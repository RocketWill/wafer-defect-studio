"""Four-image CUDA micro-overfit for grid-contrastive Spatial MIL v6."""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import numpy as np
import torch

from docs.demo.ticket31_development_corpus import (
    build_ticket31_development_corpus,
    render_ticket31_development_pixels,
)
from docs.demo.ticket31_development_gate import (
    CLASS_CODES,
    _canonical,
    _materialize_bundle,
    _score_source_model,
)
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.grid_geometry import annotation_grids
from wafer_defect_studio.model_registry import create_resnet18_spatial_logits_v5
from wafer_defect_studio.spatial_mil import (
    dense_absent_class_loss,
    overlap_consistency_loss,
    positive_spatial_topk_loss,
    present_sparse_budget_loss,
    same_image_grid_ranking_loss,
)
from wafer_defect_studio.training_input_bundle import NormalizationBounds
from wafer_defect_studio.training_patch_dataset import TrainingPatchDataset
from wafer_defect_studio.training_protocol import TrainingConfig


SCHEMA = "ticket33-micro-overfit.v1"


def run_micro_overfit(output_root: Path, *, epochs: int = 30) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("Ticket 33 micro-overfit requires CUDA")
    cases = tuple(
        case
        for case in build_ticket31_development_corpus()
        if case.seed == 101 and case.split == "train" and case.filename.endswith(("00.png", "01.png", "02.png", "03.png"))
    )
    model, losses = train_grid_contrastive_model(cases, output_root, epochs=epochs, seed=101)

    bounds = NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 0.0, 100.0)
    maps = tuple(
        _score_source_model(model, bounds, render_ticket31_development_pixels(case))
        for case in cases
    )
    return {
        "schema": SCHEMA,
        "seed": 101,
        "source_split": "train",
        "compositions": [case.composition for case in cases],
        "epochs": epochs,
        "loss": {"first": losses[0], "last": losses[-1]},
        "per_class": _separation_rows(cases, maps),
        "device": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "claims": ["micro-overfit only", "approximate localization", "not segmentation", "not Neurocle equivalence"],
    }


def train_grid_contrastive_model(cases, output_root: Path, *, epochs: int, seed: int):
    torch.manual_seed(seed)
    config = TrainingConfig(
        "ticket33-grid-contrastive", f"ticket33-seed-{seed}", len(CLASS_CODES),
        30, 2, device="cuda", seed=seed, learning_rate=0.0003,
        weights_policy="imagenet", patch_size=128, patch_stride=64,
        training_policy="spatial_mil_v5",
    )
    materialized = _materialize_bundle(cases, config, output_root / "sources")
    dataset = TrainingPatchDataset(materialized.bundle, "train", include_patch_rects=True)
    pairs = _training_pairs(materialized.bundle)
    model = create_resnet18_spatial_logits_v5(
        len(CLASS_CODES), weights="imagenet", device="cuda"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for first, second, image_id in pairs:
            items = (dataset[first], dataset[second])
            inputs = torch.stack([item[0] for item in items]).cuda()
            targets = torch.stack([item[1] for item in items]).cuda()
            rects = torch.stack([item[2] for item in items])
            batch_size, patch_count, channels, height, width = inputs.shape
            raw_logits = model(inputs.reshape(batch_size * patch_count, channels, height, width))
            logits = raw_logits.reshape(
                batch_size, patch_count, len(CLASS_CODES),
                raw_logits.shape[-2], raw_logits.shape[-1],
            )
            positive = positive_spatial_topk_loss(logits, targets)
            absent = dense_absent_class_loss(logits, targets)
            ranking = same_image_grid_ranking_loss(
                logits, targets, (image_id, image_id), margin=0.5
            )
            sparse = present_sparse_budget_loss(logits, targets)
            overlap = torch.stack([
                overlap_consistency_loss(
                    bag_logits,
                    [Rect(*map(int, values)) for values in bag_rects.tolist()],
                    feature_stride=2,
                )
                for bag_logits, bag_rects in zip(logits, rects, strict=True)
            ]).mean()
            loss = positive + absent + ranking + 0.25 * sparse + 0.10 * overlap
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(sum(epoch_losses) / len(epoch_losses))
        print(f"Ticket 33 seed {seed}: epoch {epoch + 1}/{epochs} loss={losses[-1]:.6f}", flush=True)

    return model, losses


def _training_pairs(bundle):
    pairs = []
    bags = tuple(bag for bag in bundle.patch_bags if any(source.image_asset_id == bag.image_asset_id and source.split == "train" for source in bundle.sources))
    for image_id in dict.fromkeys(bag.image_asset_id for bag in bags):
        indexed = [(index, bag) for index, bag in enumerate(bags) if bag.image_asset_id == image_id]
        normal = [index for index, bag in indexed if not bag.class_codes]
        asserted = [index for index, bag in indexed if bag.class_codes]
        if asserted:
            pairs.extend((index, normal[offset % len(normal)], image_id) for offset, index in enumerate(asserted))
        else:
            pairs.append((normal[0], normal[1], image_id))
    return tuple(pairs)


def _separation_rows(cases, maps):
    grids = annotation_grids(1536, 1536, 512, 512)
    rows = {}
    for class_index, code in enumerate(CLASS_CODES):
        asserted_scores = []
        normal_scores = []
        asserted_values = []
        for case, confidence_map in zip(cases, maps, strict=True):
            truth = case.oracle.grid_truth(grids)
            for grid in grids:
                values = confidence_map[
                    grid.y : grid.y + grid.height,
                    grid.x : grid.x + grid.width,
                    class_index,
                ].reshape(-1)
                count = max(1, ceil(values.size * 0.01))
                score = float(np.partition(values, values.size - count)[-count:].mean())
                if code in truth[(grid.row, grid.column)]:
                    asserted_scores.append(score)
                    asserted_values.append(values)
                elif not truth[(grid.row, grid.column)]:
                    normal_scores.append(score)
        weakest, hardest = min(asserted_scores), max(normal_scores)
        threshold = (weakest + hardest) / 2
        occupancies = sorted(float((values >= threshold).mean()) for values in asserted_values)
        rows[code] = {
            "weakest_asserted_grid_top_score": weakest,
            "hardest_normal_grid_top_score": hardest,
            "margin": weakest - hardest,
            "threshold": threshold,
            "asserted_grid_occupancy_p95": occupancies[ceil(0.95 * len(occupancies)) - 1],
        }
    return rows


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    report = run_micro_overfit(args.output_root.resolve(), epochs=args.epochs)
    args.report.resolve().write_text(_canonical(report), encoding="utf-8")
    print("Ticket 33 micro-overfit complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "run_micro_overfit", "train_grid_contrastive_model"]
