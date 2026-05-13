"""Deterministic mining of oracle-Normal Grid false-positive Patch Bags."""

from __future__ import annotations

import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy

from docs.demo.wafer_quality_evidence import WaferEvidenceCase
from wafer_defect_studio.training_input_bundle import TrainingInputBundle, TrainingPatchBag


SCHEMA = "spatial-hard-negative-selection.v1"
SCORE_DOMAIN = "absolute_spatial_probability"
SOURCE_SPLIT = "train"
CANDIDATE_POLICY = "oracle_truth_empty_grid; finite_map>=class_threshold; peak=max_finite_grid_value"
ORDER_POLICY = "per_class=(-peak,filename,row,column,bag_id); class_order_round_robin; global_unique_bag_cap"


def select_hard_negative_bags(
    cases: Sequence[WaferEvidenceCase],
    bundle: TrainingInputBundle,
    class_codes: Sequence[str],
    thresholds: Mapping[str, float],
    *,
    max_bags: int,
) -> dict[str, object]:
    codes = tuple(class_codes)
    if bundle.version != 2:
        raise ValueError(f"hard-negative selection requires v2 bundle, got v{bundle.version}")
    if bundle.class_codes != codes:
        raise ValueError("hard-negative selection bundle class order does not match requested classes")
    if isinstance(max_bags, bool) or not isinstance(max_bags, int) or max_bags < 0:
        raise ValueError("hard-negative selection max_bags must be a non-negative integer")
    threshold_values = _thresholds(thresholds, codes)

    source_by_filename = {}
    for source in bundle.sources:
        if source.split != SOURCE_SPLIT:
            continue
        filename = Path(source.path.replace("\\", "/")).name
        if filename in source_by_filename:
            raise ValueError(f"{filename}: multiple train sources share the same basename")
        source_by_filename[filename] = source
    bag_by_grid = {(bag.image_asset_id, bag.row, bag.column): bag for bag in bundle.patch_bags}
    candidates = {code: [] for code in codes}
    triggers_by_bag: dict[str, dict[str, object]] = {}

    for case in cases:
        if case.split != SOURCE_SPLIT:
            raise ValueError(f"{case.filename}: hard-negative case split must be train, got {case.split!r}")
        source = source_by_filename.get(Path(case.filename.replace("\\", "/")).name)
        if source is None:
            raise ValueError(f"{case.filename}: no matching train source basename in bundle")
        maps = numpy.asarray(case.absolute_maps)
        expected_shape = (case.oracle.image_height, case.oracle.image_width, len(codes))
        if maps.shape != expected_shape:
            raise ValueError(f"{case.filename}: map shape expected {expected_shape}, actual {maps.shape}")
        truth = case.oracle.grid_truth(case.grids)
        for grid in case.grids:
            bag = bag_by_grid.get((source.image_asset_id, grid.row, grid.column))
            if bag is None:
                raise ValueError(
                    f"{case.filename}: no v2 bag for row {grid.row}, column {grid.column}"
                )
            if truth[(grid.row, grid.column)]:
                continue
            left, top = max(0, grid.x), max(0, grid.y)
            right = min(case.oracle.image_width, grid.x + grid.width)
            bottom = min(case.oracle.image_height, grid.y + grid.height)
            if left >= right or top >= bottom:
                continue
            peaks = {}
            for class_index, code in enumerate(codes):
                values = maps[top:bottom, left:right, class_index]
                finite = values[numpy.isfinite(values)]
                if finite.size == 0:
                    continue
                peak = float(finite.max())
                if peak >= threshold_values[code]:
                    peaks[code] = peak
                    candidates[code].append(
                        (-peak, case.filename, grid.row, grid.column, bag.bag_id)
                    )
            if peaks:
                triggers_by_bag[bag.bag_id] = {
                    "bag_id": bag.bag_id,
                    "filename": case.filename,
                    "row": grid.row,
                    "column": grid.column,
                    "trigger_classes": [code for code in codes if code in peaks],
                    "peaks": {code: peaks[code] for code in codes if code in peaks},
                }

    for entries in candidates.values():
        entries.sort()
    selected_ids = _round_robin(candidates, codes, max_bags)
    selected = [triggers_by_bag[bag_id] for bag_id in selected_ids]
    return {
        "schema": SCHEMA,
        "score_domain": SCORE_DOMAIN,
        "source_split": SOURCE_SPLIT,
        "class_codes": list(codes),
        "thresholds": threshold_values,
        "candidate_policy": CANDIDATE_POLICY,
        "order_policy": ORDER_POLICY,
        "max_bags": max_bags,
        "counts": {
            "candidates_by_class": {code: len(candidates[code]) for code in codes},
            "unique_candidate_bags": len(triggers_by_bag),
            "selected_bags": len(selected),
        },
        "selected": selected,
    }


def resolve_hard_negative_selection(
    payload: Mapping[str, object], class_codes: Sequence[str], max_bags: int
) -> tuple[str, ...]:
    expected_keys = {
        "schema", "score_domain", "source_split", "class_codes", "thresholds",
        "candidate_policy", "order_policy", "max_bags", "counts", "selected",
    }
    if set(payload) != expected_keys or payload.get("schema") != SCHEMA:
        raise ValueError(f"hard-negative selection schema must be exact {SCHEMA}")
    if payload.get("score_domain") != SCORE_DOMAIN:
        raise ValueError(f"hard-negative selection score domain must be {SCORE_DOMAIN}")
    if payload.get("source_split") != SOURCE_SPLIT:
        raise ValueError("hard-negative selection source split must be train")
    codes = tuple(class_codes)
    if payload.get("class_codes") != list(codes):
        raise ValueError("hard-negative selection class order does not match requested classes")
    if payload.get("candidate_policy") != CANDIDATE_POLICY:
        raise ValueError("hard-negative selection candidate policy does not match contract")
    if payload.get("order_policy") != ORDER_POLICY:
        raise ValueError("hard-negative selection order policy does not match contract")
    if payload.get("max_bags") != max_bags:
        raise ValueError("hard-negative selection max_bags does not match requested cap")
    _thresholds(payload.get("thresholds"), codes)
    counts = payload.get("counts")
    selected = payload.get("selected")
    if not isinstance(counts, Mapping) or set(counts) != {"candidates_by_class", "unique_candidate_bags", "selected_bags"}:
        raise ValueError("hard-negative selection counts must be complete")
    per_class = counts["candidates_by_class"]
    if not isinstance(per_class, Mapping) or set(per_class) != set(codes) or any(type(value) is not int or value < 0 for value in per_class.values()):
        raise ValueError("hard-negative selection candidate counts must match class order")
    if type(counts["selected_bags"]) is not int or not isinstance(selected, list) or counts["selected_bags"] != len(selected) or len(selected) > max_bags:
        raise ValueError("hard-negative selection selected count does not match artifact")
    bag_ids = []
    for item in selected:
        if not isinstance(item, Mapping) or set(item) != {"bag_id", "filename", "row", "column", "trigger_classes", "peaks"}:
            raise ValueError("hard-negative selection selected entry is invalid")
        triggers = item["trigger_classes"]
        peaks = item["peaks"]
        if not isinstance(triggers, list) or triggers != [code for code in codes if code in triggers] or not triggers:
            raise ValueError(f"{item.get('bag_id')}: trigger class order is invalid")
        if (
            not isinstance(item["bag_id"], str)
            or not isinstance(item["filename"], str)
            or type(item["row"]) is not int
            or type(item["column"]) is not int
        ):
            raise ValueError("hard-negative selection selected bag context is invalid")
        if not isinstance(peaks, Mapping) or set(peaks) != set(triggers) or any(isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) for value in peaks.values()):
            raise ValueError(f"{item.get('bag_id')}: trigger peaks are invalid")
        for code in triggers:
            if peaks[code] < payload["thresholds"][code]:
                raise ValueError(f"{item['bag_id']}: {code} trigger peak is below artifact threshold")
        bag_ids.append(item["bag_id"])
    if len(set(bag_ids)) != len(bag_ids):
        raise ValueError("hard-negative selection selected bags must be globally unique")
    if type(counts["unique_candidate_bags"]) is not int or counts["unique_candidate_bags"] < len(selected):
        raise ValueError("hard-negative selection unique candidate count is invalid")
    return tuple(bag_ids)


def hard_negative_selection_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _thresholds(values: object, codes: Sequence[str]) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(codes):
        raise ValueError("hard-negative selection thresholds must match class order")
    result = {}
    for code in codes:
        value = values[code]
        if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{code}: hard-negative threshold must be finite in [0, 1]")
        result[code] = float(value)
    return result


def _round_robin(candidates: Mapping[str, list[tuple]], codes: Sequence[str], cap: int) -> list[str]:
    positions = {code: 0 for code in codes}
    selected = []
    seen = set()
    while len(selected) < cap:
        advanced = False
        for code in codes:
            entries = candidates[code]
            while positions[code] < len(entries):
                bag_id = entries[positions[code]][-1]
                positions[code] += 1
                advanced = True
                if bag_id not in seen:
                    seen.add(bag_id)
                    selected.append(bag_id)
                    break
            if len(selected) == cap:
                break
        if not advanced:
            break
    return selected
