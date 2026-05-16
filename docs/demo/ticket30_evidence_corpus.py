"""Frozen generated held-out corpus for Ticket 30 matched evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch

SEEDS = (17, 42, 91)
CLASS_CODES = ("scratch", "particle")
FROZEN_CORPUS_SHA256 = "e733096ecf3bc52970ea88bb20d1d2dbeea36c6a64ed3fcf15ee98467c6cc0b8"


@dataclass(frozen=True)
class EvidenceInstance:
    instance_id: str
    defect: Scratch | Particle


@dataclass(frozen=True)
class EvidenceCase:
    seed: int
    filename: str
    split: str
    instances: tuple[EvidenceInstance, ...]
    oracle: DefectOracle
    pixel_sha256: str


def build_ticket30_evidence_corpus() -> tuple[EvidenceCase, ...]:
    cases = []
    for seed_index, seed in enumerate(SEEDS):
        scratches = tuple(
            EvidenceInstance(
                f"ticket30-{seed}-scratch-{index:03d}",
                Scratch(
                    "scratch",
                    ((20 + index % 15 * 32, 20 + index // 15 * 45 + seed_index),
                     (40 + index % 15 * 32, 20 + index // 15 * 45 + seed_index)),
                    2,
                ),
            )
            for index in range(150)
        )
        particles = tuple(
            EvidenceInstance(
                f"ticket30-{seed}-particle-{index:03d}",
                Particle(
                    "particle",
                    (1044 + index % 15 * 32, 1044 + index // 15 * 45 + seed_index),
                    3,
                ),
            )
            for index in range(150)
        )
        instances = scratches + particles
        oracle = DefectOracle(1536, 1536, tuple(instance.defect for instance in instances))
        cases.append(EvidenceCase(
            seed, f"ticket30-evidence-{seed}.png", "test", instances, oracle,
            _pixel_sha256(render_ticket30_evidence_pixels(oracle)),
        ))
    return tuple(cases)


def serialize_ticket30_evidence_corpus(corpus: tuple[EvidenceCase, ...]) -> str:
    payload = {
        "schema_version": 1,
        "seeds": list(SEEDS),
        "class_codes": list(CLASS_CODES),
        "cases": [
            {
                "seed": case.seed,
                "filename": case.filename,
                "split": case.split,
                "image_width": case.oracle.image_width,
                "image_height": case.oracle.image_height,
                "pixel_sha256": case.pixel_sha256,
                "instances": [
                    {
                        "instance_id": instance.instance_id,
                        "defect": {
                            "kind": "scratch" if isinstance(instance.defect, Scratch) else "particle",
                            **asdict(instance.defect),
                        },
                    }
                    for instance in case.instances
                ],
            }
            for case in corpus
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def hash_ticket30_evidence_corpus(corpus: tuple[EvidenceCase, ...]) -> str:
    return hashlib.sha256(serialize_ticket30_evidence_corpus(corpus).encode("utf-8")).hexdigest()


def render_ticket30_evidence_pixels(oracle: DefectOracle) -> np.ndarray:
    """Render deterministic grayscale pixels directly from the frozen oracle."""

    pixels = np.full((oracle.image_height, oracle.image_width), 128, dtype=np.uint8)
    for defect in oracle.defects:
        if isinstance(defect, Scratch):
            for start, end in zip(defect.points, defect.points[1:]):
                _paint_segment(pixels, start, end, defect.radius, 24)
        else:
            _paint_disk(pixels, defect.center, defect.radius, 232)
    return pixels


def _pixel_sha256(pixels: np.ndarray) -> str:
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def _paint_disk(pixels: np.ndarray, center: tuple[int, int], radius: int, value: int) -> None:
    x, y = center
    top, bottom = max(0, y - radius), min(pixels.shape[0] - 1, y + radius)
    left, right = max(0, x - radius), min(pixels.shape[1] - 1, x + radius)
    yy, xx = np.ogrid[top : bottom + 1, left : right + 1]
    region = pixels[top : bottom + 1, left : right + 1]
    region[(xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2] = value


def _paint_segment(
    pixels: np.ndarray, start: tuple[int, int], end: tuple[int, int], radius: int, value: int
) -> None:
    x0, y0 = start
    x1, y1 = end
    top, bottom = max(0, min(y0, y1) - radius), min(pixels.shape[0] - 1, max(y0, y1) + radius)
    left, right = max(0, min(x0, x1) - radius), min(pixels.shape[1] - 1, max(x0, x1) + radius)
    yy, xx = np.ogrid[top : bottom + 1, left : right + 1]
    dx, dy = x1 - x0, y1 - y0
    t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / (dx * dx + dy * dy), 0, 1)
    region = pixels[top : bottom + 1, left : right + 1]
    region[(xx - (x0 + t * dx)) ** 2 + (yy - (y0 + t * dy)) ** 2 <= radius ** 2] = value


def resolve_ticket30_evidence_corpus(serialized: str, expected_sha256: str) -> tuple[EvidenceCase, ...]:
    actual_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"ticket30 evidence corpus SHA-256 mismatch: expected={expected_sha256} actual={actual_sha256}"
        )
    try:
        payload = json.loads(serialized)
        cases = tuple(_case_from_payload(case) for case in payload["cases"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid ticket30 evidence corpus: {error}") from error
    _validate(payload, cases)
    return cases


def _case_from_payload(payload: dict) -> EvidenceCase:
    instances = []
    for value in payload["instances"]:
        defect = dict(value["defect"])
        kind = defect.pop("kind")
        if kind == "scratch":
            defect["points"] = tuple(tuple(point) for point in defect["points"])
            resolved = Scratch(**defect)
        elif kind == "particle":
            defect["center"] = tuple(defect["center"])
            resolved = Particle(**defect)
        else:
            raise ValueError(f"unknown ticket30 evidence defect kind: {kind}")
        instances.append(EvidenceInstance(value["instance_id"], resolved))
    oracle = DefectOracle(
        payload["image_width"], payload["image_height"], tuple(value.defect for value in instances)
    )
    return EvidenceCase(
        payload["seed"], payload["filename"], payload["split"], tuple(instances), oracle,
        payload["pixel_sha256"],
    )


def _validate(payload: dict, cases: tuple[EvidenceCase, ...]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError(f"ticket30 evidence corpus schema drift: {payload.get('schema_version')!r}")
    if tuple(payload.get("seeds", ())) != SEEDS or tuple(case.seed for case in cases) != SEEDS:
        raise ValueError("ticket30 evidence corpus seed membership/order drift")
    if tuple(payload.get("class_codes", ())) != CLASS_CODES:
        raise ValueError("ticket30 evidence corpus class order/membership drift")
    if any(case.split != "test" for case in cases):
        raise ValueError("ticket30 evidence corpus membership drift: every case must be test")
    instances = tuple(instance for case in cases for instance in case.instances)
    actual_classes = tuple(dict.fromkeys(instance.defect.class_code for instance in instances))
    if actual_classes != CLASS_CODES:
        raise ValueError(f"ticket30 evidence corpus class order/membership drift: actual={actual_classes!r}")
    for case in cases:
        counts = {
            code: sum(instance.defect.class_code == code for instance in case.instances)
            for code in CLASS_CODES
        }
        if counts != {"scratch": 150, "particle": 150}:
            raise ValueError(
                f"ticket30 evidence corpus instance count drift: seed={case.seed} actual={counts!r}"
            )
        actual_pixel_sha256 = _pixel_sha256(render_ticket30_evidence_pixels(case.oracle))
        if actual_pixel_sha256 != case.pixel_sha256:
            raise ValueError(
                "ticket30 evidence pixel SHA-256 mismatch: "
                f"expected={case.pixel_sha256} actual={actual_pixel_sha256}"
            )
    instance_ids = tuple(instance.instance_id for instance in instances)
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("ticket30 evidence corpus duplicate instance_id")
