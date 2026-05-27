"""Position-diverse generated development cases for Ticket 31."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch
from docs.demo.ticket30_evidence_corpus import render_ticket30_evidence_pixels
from docs.demo.ticket31_contract import DEVELOPMENT_SEEDS
from wafer_defect_studio.grid_geometry import annotation_grids


@dataclass(frozen=True)
class DevelopmentInstance:
    instance_id: str
    defect: Scratch | Particle


@dataclass(frozen=True)
class DevelopmentCase:
    seed: int
    filename: str
    family: str
    split: str
    instances: tuple[DevelopmentInstance, ...]
    oracle: DefectOracle
    composition: str
    scratch_count: int
    particle_count: int
    position_bin: str
    scratch_orientation: str | None
    scratch_length: int | None
    particle_radius: int | None
    contrast_bin: str
    background_index: int
    pixel_sha256: str


_POSITIONS = {"near": 96, "center": 256, "far": 416}
_ORIENTATIONS = ("horizontal", "vertical", "diagonal")
_LENGTHS = (24, 48, 80)
_RADII = (2, 3, 5)
_CONTRASTS = {
    "low": (96, 160),
    "medium": (64, 192),
    "high": (24, 232),
}


def build_ticket31_development_corpus() -> tuple[DevelopmentCase, ...]:
    cases = []
    for seed_index, seed in enumerate(DEVELOPMENT_SEEDS):
        for split_index, split in enumerate(("train", "validation")):
            family = f"ticket31-{split}-{seed}"
            for index in range(24):
                composition = ("normal", "scratch", "particle", "both")[index % 4]
                variant = index // 4
                scratch_count = (
                    1 + variant % 2 if composition in {"scratch", "both"} else 0
                )
                particle_count = (
                    1 + variant % 2 if composition in {"particle", "both"} else 0
                )
                position_bin = tuple(_POSITIONS)[(index + seed_index) % 3]
                orientation = _ORIENTATIONS[(index + split_index) % 3]
                length = _LENGTHS[(index + seed_index + split_index) % 3]
                radius = _RADII[(index // 3 + seed_index + split_index) % 3]
                contrast_bin = tuple(_CONTRASTS)[(index + 2 * seed_index + split_index) % 3]
                prefix = f"ticket31-{seed}-{split}-{index:02d}"
                instances = []
                for instance_index in range(scratch_count):
                    grid = divmod(
                        (variant + (3 if composition == "both" else 0)
                         + 3 * instance_index + seed_index) % 9,
                        3,
                    )
                    defect = _scratch(
                        grid, position_bin,
                        _ORIENTATIONS[(index + split_index + instance_index) % 3],
                        _LENGTHS[(index + seed_index + split_index + instance_index) % 3],
                    )
                    instances.append(DevelopmentInstance(
                        f"{prefix}-scratch-{instance_index}", defect
                    ))
                for instance_index in range(particle_count):
                    grid = divmod(
                        (variant + (3 if composition == "both" else 0) + 4
                         + 2 * instance_index + seed_index + split_index) % 9,
                        3,
                    )
                    defect = _particle(
                        grid, position_bin,
                        _RADII[(index // 3 + seed_index + split_index + instance_index) % 3],
                    )
                    instances.append(DevelopmentInstance(
                        f"{prefix}-particle-{instance_index}", defect
                    ))
                instances = tuple(instances)
                oracle = DefectOracle(
                    1536, 1536, tuple(instance.defect for instance in instances)
                )
                case = DevelopmentCase(
                    seed=seed,
                    filename=f"{prefix}.png",
                    family=family,
                    split=split,
                    instances=instances,
                    oracle=oracle,
                    composition=composition,
                    scratch_count=scratch_count,
                    particle_count=particle_count,
                    position_bin=position_bin,
                    scratch_orientation=orientation if scratch_count else None,
                    scratch_length=length if scratch_count else None,
                    particle_radius=radius if particle_count else None,
                    contrast_bin=contrast_bin,
                    background_index=seed_index * 29 + split_index * 13 + index,
                    pixel_sha256="",
                )
                pixels = render_ticket31_development_pixels(case)
                cases.append(replace(case, pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
    return tuple(cases)


def development_grid_annotations(
    case: DevelopmentCase,
) -> dict[tuple[int, int], tuple[str, ...]]:
    grids = annotation_grids(1536, 1536, 512, 512)
    return {
        grid: codes
        for grid, codes in case.oracle.grid_truth(grids).items()
        if codes
    }


def render_ticket31_development_pixels(case: DevelopmentCase) -> np.ndarray:
    yy, xx = np.indices((1536, 1536), dtype=np.uint16)
    background = 104 + (
        (xx // 64) * 3 + (yy // 64) * 5 + case.background_index
    ) % 40
    pixels = background.astype(np.uint8)
    masks = render_ticket30_evidence_pixels(case.oracle)
    scratch_value, particle_value = _CONTRASTS[case.contrast_bin]
    pixels[masks == 24] = scratch_value
    pixels[masks == 232] = particle_value
    return pixels


def _scratch(
    grid: tuple[int, int], position_bin: str, orientation: str, length: int
) -> Scratch:
    row, column = grid
    center_x = column * 512 + _POSITIONS[position_bin]
    center_y = row * 512 + _POSITIONS[position_bin]
    half = length // 2
    if orientation == "horizontal":
        points = ((center_x - half, center_y), (center_x + half, center_y))
    elif orientation == "vertical":
        points = ((center_x, center_y - half), (center_x, center_y + half))
    else:
        points = ((center_x - half, center_y - half), (center_x + half, center_y + half))
    return Scratch("scratch", points, 3)


def _particle(grid: tuple[int, int], position_bin: str, radius: int) -> Particle:
    row, column = grid
    offset = _POSITIONS[position_bin]
    return Particle("particle", (column * 512 + offset, row * 512 + (512 - offset)), radius)


__all__ = [
    "DevelopmentCase",
    "DevelopmentInstance",
    "build_ticket31_development_corpus",
    "development_grid_annotations",
    "render_ticket31_development_pixels",
]
