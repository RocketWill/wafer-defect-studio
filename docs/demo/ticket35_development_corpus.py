"""Density-diverse, oracle-only development corpus for Ticket 35.

The builder creates deterministic case descriptors and DefectOracle truth. It
does not render or persist image pixels; callers may render one development
case on demand through :func:`render_ticket35_development_pixels`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch
from docs.demo.ticket31_contract import DEVELOPMENT_SEEDS, FINAL_MEMBERS


SCHEMA = "ticket35-density-development-corpus.v1"
IMAGE_WIDTH = 1536
IMAGE_HEIGHT = 1536
GRID_SIZE = 512
SPLITS = ("train", "validation")
COUNT_LEVELS = (1, 2, 8, 32, 150)
COMPOSITIONS = ("normal", "scratch", "particle", "both")
PLACEMENT_STRATA = ("interior", "near_boundary", "cross_grid")
SCRATCH_ORIENTATIONS = ("horizontal", "vertical", "diagonal")
SCRATCH_LENGTHS = (24, 48, 80, 128)
PARTICLE_RADII = (2, 3, 5, 8)
RESERVED_FINAL_MEMBER_IDS = (
    "ticket35-final-member-01",
    "ticket35-final-member-02",
    "ticket35-final-member-03",
)


@dataclass(frozen=True)
class DevelopmentInstance:
    instance_id: str
    defect: Scratch | Particle


@dataclass(frozen=True)
class DevelopmentCase:
    seed: int
    split: str
    family: str
    case_id: str
    filename: str
    composition: str
    scratch_count: int
    particle_count: int
    placement_stratum: str
    scratch_orientation: str | None
    scratch_length: int | None
    particle_radius: int | None
    instances: tuple[DevelopmentInstance, ...]
    oracle: DefectOracle


def build_ticket35_development_corpus() -> tuple[DevelopmentCase, ...]:
    """Build deterministic development cases without materializing pixels."""

    cases: list[DevelopmentCase] = []
    for seed_index, seed in enumerate(DEVELOPMENT_SEEDS):
        for split_index, split in enumerate(SPLITS):
            specs = [("normal", 0, 0, "interior", 0)]
            specs.extend(
                (
                    "scratch",
                    count,
                    0,
                    _placement(seed_index, split_index, index),
                    index,
                )
                for index, count in enumerate(COUNT_LEVELS, start=1)
            )
            specs.extend(
                (
                    "particle",
                    0,
                    count,
                    _placement(seed_index, split_index, index + 5),
                    index + 5,
                )
                for index, count in enumerate(COUNT_LEVELS, start=1)
            )
            specs.append(("both", 8, 8, "cross_grid", 11))
            for case_index, (composition, scratch_count, particle_count, placement, variant) in enumerate(specs):
                case_id = (
                    f"ticket35-density-{seed}-{split}-{composition}-"
                    f"{scratch_count:03d}-{particle_count:03d}-{case_index:02d}"
                )
                instances = _build_instances(
                    seed_index,
                    split_index,
                    case_id,
                    scratch_count,
                    particle_count,
                    placement,
                    variant,
                )
                cases.append(
                    DevelopmentCase(
                        seed=seed,
                        split=split,
                        family=f"{case_id}-family",
                        case_id=case_id,
                        filename=f"{case_id}.png",
                        composition=composition,
                        scratch_count=scratch_count,
                        particle_count=particle_count,
                        placement_stratum=placement,
                        scratch_orientation=(
                            _scratch_orientation(seed_index, split_index, variant)
                            if scratch_count
                            else None
                        ),
                        scratch_length=(
                            _scratch_length(seed_index, split_index, variant)
                            if scratch_count
                            else None
                        ),
                        particle_radius=(
                            _particle_radius(seed_index, split_index, variant)
                            if particle_count
                            else None
                        ),
                        instances=instances,
                        oracle=DefectOracle(
                            IMAGE_WIDTH,
                            IMAGE_HEIGHT,
                            tuple(instance.defect for instance in instances),
                        ),
                    )
                )
    _validate_corpus(cases)
    return tuple(cases)


def build_ticket35_development_metadata(
    corpus: Sequence[DevelopmentCase] | None = None,
) -> dict[str, object]:
    """Build canonical tracked metadata for the oracle-only corpus."""

    cases = tuple(build_ticket35_development_corpus() if corpus is None else corpus)
    _validate_corpus(cases)
    return {
        "schema": SCHEMA,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "splits": list(SPLITS),
        "count_levels": list(COUNT_LEVELS),
        "compositions": list(COMPOSITIONS),
        "placement_strata": list(PLACEMENT_STRATA),
        "scratch_orientations": list(SCRATCH_ORIENTATIONS),
        "scratch_lengths": list(SCRATCH_LENGTHS),
        "particle_radii": list(PARTICLE_RADII),
        "case_count": len(cases),
        "instance_count": sum(len(case.instances) for case in cases),
        "pixels_materialized": False,
        "reserved_final_member_ids": list(RESERVED_FINAL_MEMBER_IDS),
        "reserved_final_status": "unopened_no_pixels_materialized",
        "promotion_use": "later_only",
        "reserved_final": {
            "reserved_final_member_ids": list(RESERVED_FINAL_MEMBER_IDS),
            "status": "unopened_no_pixels_materialized",
            "promotion_use": "later_only",
        },
        "cases": [_metadata_case(case) for case in cases],
        "corpus_sha256": hash_ticket35_development_corpus(cases),
    }


def serialize_ticket35_development_corpus(
    corpus: Sequence[DevelopmentCase],
) -> str:
    """Serialize complete oracle descriptors for content addressing."""

    payload = {
        "schema": SCHEMA,
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "splits": list(SPLITS),
        "count_levels": list(COUNT_LEVELS),
        "cases": [_serialize_case(case) for case in corpus],
    }
    return _canonical_json(payload)


def hash_ticket35_development_corpus(
    corpus: Sequence[DevelopmentCase],
) -> str:
    """Return the SHA-256 of the canonical complete corpus descriptors."""

    return hashlib.sha256(
        serialize_ticket35_development_corpus(corpus).encode("utf-8")
    ).hexdigest()


def canonical_ticket35_development_metadata_json(
    metadata: Mapping[str, object],
) -> str:
    return _canonical_json(metadata)


def verify_ticket35_development_metadata(
    metadata: Mapping[str, object],
    corpus: Sequence[DevelopmentCase] | None = None,
) -> None:
    """Verify canonical metadata and its content-addressed build hash."""

    cases = tuple(build_ticket35_development_corpus() if corpus is None else corpus)
    expected = build_ticket35_development_metadata(cases)
    actual_hash = metadata.get("corpus_sha256") if isinstance(metadata, Mapping) else None
    if actual_hash != expected["corpus_sha256"]:
        raise ValueError(
            "Ticket 35 development corpus SHA-256 mismatch: "
            f"expected={expected['corpus_sha256']} actual={actual_hash}"
        )
    if dict(metadata) != expected:
        raise ValueError("Ticket 35 development corpus metadata drift")


def render_ticket35_development_pixels(case: DevelopmentCase):
    """Render one development oracle on demand; the builder never calls this."""

    from docs.demo.ticket30_evidence_corpus import render_ticket30_evidence_pixels

    return render_ticket30_evidence_pixels(case.oracle)


def _build_instances(
    seed_index: int,
    split_index: int,
    case_id: str,
    scratch_count: int,
    particle_count: int,
    placement: str,
    variant: int,
) -> tuple[DevelopmentInstance, ...]:
    instances: list[DevelopmentInstance] = []
    for index in range(scratch_count):
        orientation = _scratch_orientation(seed_index, split_index, variant + index)
        length = _scratch_length(seed_index, split_index, variant + index)
        defect = _scratch(
            seed_index,
            split_index,
            variant + index,
            placement,
            orientation,
            length,
        )
        instances.append(DevelopmentInstance(f"{case_id}-scratch-{index:03d}", defect))
    for index in range(particle_count):
        radius = _particle_radius(seed_index, split_index, variant + index)
        defect = _particle(
            seed_index,
            split_index,
            variant + index,
            placement,
            radius,
        )
        instances.append(DevelopmentInstance(f"{case_id}-particle-{index:03d}", defect))
    return tuple(instances)


def _scratch(
    seed_index: int,
    split_index: int,
    index: int,
    placement: str,
    orientation: str,
    length: int,
) -> Scratch:
    row, column = _grid(seed_index, split_index, index, 0)
    half = length // 2
    if placement == "cross_grid":
        boundary_x = (column + 1) * GRID_SIZE if column < 2 else column * GRID_SIZE
        boundary_y = (row + 1) * GRID_SIZE if row < 2 else row * GRID_SIZE
        if orientation == "vertical":
            center_x, center_y = column * GRID_SIZE + 160, boundary_y
            points = ((center_x, center_y - half), (center_x, center_y + half))
        elif orientation == "diagonal":
            center_x, center_y = boundary_x, row * GRID_SIZE + 256
            points = ((center_x - half, center_y - half), (center_x + half, center_y + half))
        else:
            center_x, center_y = boundary_x, row * GRID_SIZE + 256
            points = ((center_x - half, center_y), (center_x + half, center_y))
    elif placement == "near_boundary":
        boundary_x = (column + 1) * GRID_SIZE if column < 2 else column * GRID_SIZE
        boundary_y = (row + 1) * GRID_SIZE if row < 2 else row * GRID_SIZE
        offset = 6 + (index % 5)
        if orientation == "vertical":
            center_x, center_y = column * GRID_SIZE + 256, boundary_y - offset
            points = ((center_x, center_y - half), (center_x, center_y + half))
        elif orientation == "diagonal":
            center_x, center_y = boundary_x - offset, row * GRID_SIZE + 256
            points = ((center_x - half, center_y - half), (center_x + half, center_y + half))
        else:
            center_x, center_y = boundary_x - offset, row * GRID_SIZE + 256
            points = ((center_x - half, center_y), (center_x + half, center_y))
    else:
        row, column = _grid(seed_index, split_index, index, 3)
        center_x = column * GRID_SIZE + 256 + ((index % 5) - 2) * 20
        center_y = row * GRID_SIZE + 256 + ((index % 7) - 3) * 12
        if orientation == "vertical":
            points = ((center_x, center_y - half), (center_x, center_y + half))
        elif orientation == "diagonal":
            points = ((center_x - half, center_y - half), (center_x + half, center_y + half))
        else:
            points = ((center_x - half, center_y), (center_x + half, center_y))
    return Scratch("scratch", points, 3)


def _particle(
    seed_index: int,
    split_index: int,
    index: int,
    placement: str,
    radius: int,
) -> Particle:
    row, column = _grid(seed_index, split_index, index, 7)
    if placement == "cross_grid":
        if index % 2:
            boundary = (column + 1) * GRID_SIZE if column < 2 else column * GRID_SIZE
            center = (boundary, row * GRID_SIZE + 96 + (index % 7) * 48)
        else:
            boundary = (row + 1) * GRID_SIZE if row < 2 else row * GRID_SIZE
            center = (column * GRID_SIZE + 96 + (index % 7) * 48, boundary)
    elif placement == "near_boundary":
        boundary = (column + 1) * GRID_SIZE if column < 2 else column * GRID_SIZE
        offset = radius + 1 + (index % 4)
        center = (boundary - offset, row * GRID_SIZE + 96 + (index % 7) * 48)
    else:
        center = (
            column * GRID_SIZE + 256 + ((index % 5) - 2) * 20,
            row * GRID_SIZE + 256 + ((index % 7) - 3) * 12,
        )
    return Particle("particle", center, radius)


def _grid(seed_index: int, split_index: int, index: int, offset: int) -> tuple[int, int]:
    value = (seed_index * 5 + split_index * 3 + index * 7 + offset) % 9
    return divmod(value, 3)


def _placement(seed_index: int, split_index: int, index: int) -> str:
    return PLACEMENT_STRATA[(seed_index + split_index + index) % len(PLACEMENT_STRATA)]


def _scratch_orientation(seed_index: int, split_index: int, index: int) -> str:
    return SCRATCH_ORIENTATIONS[(seed_index + split_index + index) % len(SCRATCH_ORIENTATIONS)]


def _scratch_length(seed_index: int, split_index: int, index: int) -> int:
    return SCRATCH_LENGTHS[(seed_index * 2 + split_index + index) % len(SCRATCH_LENGTHS)]


def _particle_radius(seed_index: int, split_index: int, index: int) -> int:
    return PARTICLE_RADII[(seed_index + split_index * 2 + index) % len(PARTICLE_RADII)]


def _metadata_case(case: DevelopmentCase) -> dict[str, object]:
    return {
        "seed": case.seed,
        "split": case.split,
        "family": case.family,
        "case_id": case.case_id,
        "filename": case.filename,
        "composition": case.composition,
        "scratch_count": case.scratch_count,
        "particle_count": case.particle_count,
        "placement_stratum": case.placement_stratum,
        "scratch_orientation": case.scratch_orientation,
        "scratch_length": case.scratch_length,
        "particle_radius": case.particle_radius,
        "instance_count": len(case.instances),
    }


def _serialize_case(case: DevelopmentCase) -> dict[str, object]:
    return {
        **_metadata_case(case),
        "image_width": case.oracle.image_width,
        "image_height": case.oracle.image_height,
        "instances": [
            {
                "instance_id": instance.instance_id,
                "defect": _defect_payload(instance.defect),
            }
            for instance in case.instances
        ],
    }


def _defect_payload(defect: Scratch | Particle) -> dict[str, object]:
    return {
        "kind": "scratch" if isinstance(defect, Scratch) else "particle",
        **asdict(defect),
    }


def _validate_corpus(corpus: Sequence[DevelopmentCase]) -> None:
    cases = tuple(corpus)
    if not cases:
        raise ValueError("Ticket 35 development corpus must not be empty")
    if tuple(dict.fromkeys(case.seed for case in cases)) != DEVELOPMENT_SEEDS:
        raise ValueError("Ticket 35 development seed membership/order drift")
    if tuple(dict.fromkeys(case.split for case in cases)) != SPLITS:
        raise ValueError("Ticket 35 development split membership/order drift")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Ticket 35 development duplicate case_id")
    families = {split: {case.family for case in cases if case.split == split} for split in SPLITS}
    if families["train"] & families["validation"]:
        raise ValueError("Ticket 35 train/validation family overlap")
    instance_ids = {
        split: {
            instance.instance_id
            for case in cases
            if case.split == split
            for instance in case.instances
        }
        for split in SPLITS
    }
    if instance_ids["train"] & instance_ids["validation"]:
        raise ValueError("Ticket 35 train/validation instance overlap")
    all_names = {
        value
        for case in cases
        for value in (case.case_id, case.family, case.filename)
    }
    all_names.update(instance.instance_id for case in cases for instance in case.instances)
    if any(any(member in value for member in FINAL_MEMBERS) for value in all_names):
        raise ValueError("Ticket 35 development corpus contains Ticket 34 final member")
    if all_names & set(RESERVED_FINAL_MEMBER_IDS):
        raise ValueError("Ticket 35 development corpus contains reserved final member")
    for seed in DEVELOPMENT_SEEDS:
        for split in SPLITS:
            selected = tuple(case for case in cases if case.seed == seed and case.split == split)
            if {case.composition for case in selected} != set(COMPOSITIONS):
                raise ValueError(f"Ticket 35 composition strata missing: seed={seed} split={split}")
            if {
                case.scratch_count
                for case in selected
                if case.composition == "scratch"
            } != set(COUNT_LEVELS):
                raise ValueError(f"Ticket 35 scratch count levels missing: seed={seed} split={split}")
            if {
                case.particle_count
                for case in selected
                if case.composition == "particle"
            } != set(COUNT_LEVELS):
                raise ValueError(f"Ticket 35 particle count levels missing: seed={seed} split={split}")


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "COMPOSITIONS",
    "COUNT_LEVELS",
    "DevelopmentCase",
    "DevelopmentInstance",
    "DEVELOPMENT_SEEDS",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "PARTICLE_RADII",
    "PLACEMENT_STRATA",
    "RESERVED_FINAL_MEMBER_IDS",
    "SCHEMA",
    "SCRATCH_LENGTHS",
    "SCRATCH_ORIENTATIONS",
    "SPLITS",
    "build_ticket35_development_corpus",
    "build_ticket35_development_metadata",
    "canonical_ticket35_development_metadata_json",
    "hash_ticket35_development_corpus",
    "render_ticket35_development_pixels",
    "serialize_ticket35_development_corpus",
    "verify_ticket35_development_metadata",
]
