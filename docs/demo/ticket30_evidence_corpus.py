"""Frozen generated held-out corpus for Ticket 30 matched evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch

SEEDS = (17, 42, 91)
CLASS_CODES = ("scratch", "particle")
FROZEN_CORPUS_SHA256 = "9a2557b59b77d1789624a3290b273e83b8512568b1b5c456fd8123114c45a3e1"


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


def build_ticket30_evidence_corpus() -> tuple[EvidenceCase, ...]:
    cases = []
    for seed_index, seed in enumerate(SEEDS):
        scratches = tuple(
            EvidenceInstance(
                f"ticket30-{seed}-scratch-{index:03d}",
                Scratch(
                    "scratch",
                    ((30 + index % 15 * 98, 30 + index // 15 * 145 + seed_index * 3),
                     (50 + index % 15 * 98, 30 + index // 15 * 145 + seed_index * 3)),
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
                    (60 + index % 15 * 96, 80 + index // 15 * 140 + seed_index * 3),
                    3,
                ),
            )
            for index in range(150)
        )
        instances = scratches + particles
        cases.append(
            EvidenceCase(
                seed,
                f"ticket30-evidence-{seed}.png",
                "test",
                instances,
                DefectOracle(1536, 1536, tuple(instance.defect for instance in instances)),
            )
        )
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
    return EvidenceCase(payload["seed"], payload["filename"], payload["split"], tuple(instances), oracle)


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
    instance_ids = tuple(instance.instance_id for instance in instances)
    if len(set(instance_ids)) != len(instance_ids):
        raise ValueError("ticket30 evidence corpus duplicate instance_id")
