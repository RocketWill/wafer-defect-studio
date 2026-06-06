"""Sparse point/scribble truth for Ticket 35 development cases.

This sidecar retains source-coordinate sparse supervision only.  It contains
one row per generated defect instance and deliberately does not materialize
pixels or claim a complete region annotation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from docs.demo.defect_oracle import Particle, Scratch
from docs.demo.ticket31_contract import FINAL_MEMBERS
from docs.demo.ticket35_development_corpus import (
    RESERVED_FINAL_MEMBER_IDS,
    DevelopmentCase,
    build_ticket35_development_corpus,
    hash_ticket35_development_corpus,
)


BUNDLE_SCHEMA = "ticket35-sparse-localization-bundle.v1"
MANIFEST_SCHEMA = "ticket35-sparse-localization-manifest.v1"
CORPUS_SHA256 = "1844b1947e478a57bf71dce729160ec4c2d3dcc68d1185b300a75db15482c229"
IMAGE_WIDTH = 1536
IMAGE_HEIGHT = 1536
EXPECTED_CASE_COUNT = 120
EXPECTED_INSTANCE_COUNT = 4020
CLASS_CODES = ("scratch", "particle")
SPLITS = ("train", "validation")
PROVENANCE = "synthetic_defect_oracle"
KIND_BY_CLASS = {"particle": "point", "scratch": "scribble"}
MANIFEST_PATH = Path(__file__).with_name("ticket35-sparse-localization-manifest.json")


Point = tuple[int, int]


@dataclass(frozen=True)
class SparseLocalizationInstance:
    """One sparse source-coordinate truth row for one development instance."""

    instance_id: str
    case_id: str
    filename: str
    split: str
    class_code: str
    kind: str
    points: tuple[Point, ...]
    provenance: str


@dataclass(frozen=True)
class SparseLocalizationBundle:
    """Content-addressed sparse supervision sidecar."""

    schema: str
    corpus_sha256: str
    instances: tuple[SparseLocalizationInstance, ...]


def build_ticket35_sparse_localization_bundle(
    corpus: Sequence[DevelopmentCase] | None = None,
) -> SparseLocalizationBundle:
    """Build the deterministic sparse sidecar from the frozen 35.02 corpus."""

    cases = tuple(build_ticket35_development_corpus() if corpus is None else corpus)
    actual_corpus_sha256 = hash_ticket35_development_corpus(cases)
    if actual_corpus_sha256 != CORPUS_SHA256:
        raise ValueError(
            "Ticket 35 sparse localization corpus SHA-256 mismatch: "
            f"expected={CORPUS_SHA256} actual={actual_corpus_sha256}"
        )
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(
            "Ticket 35 sparse localization case coverage mismatch: "
            f"expected={EXPECTED_CASE_COUNT} actual={len(cases)}"
        )
    rows = tuple(_row_from_case(case, instance) for case in cases for instance in case.instances)
    bundle = SparseLocalizationBundle(BUNDLE_SCHEMA, actual_corpus_sha256, rows)
    validate_ticket35_sparse_localization_bundle(bundle, cases)
    return bundle


def validate_ticket35_sparse_localization_bundle(
    bundle: SparseLocalizationBundle,
    corpus: Sequence[DevelopmentCase] | None = None,
) -> None:
    """Validate sparse truth against every expected corpus instance.

    Validation is intentionally strict: a missing, extra, reordered, or
    changed row fails with the affected instance and case in the error.
    """

    if not isinstance(bundle, SparseLocalizationBundle):
        raise ValueError("Ticket 35 sparse localization bundle must be a bundle object")
    cases = tuple(build_ticket35_development_corpus() if corpus is None else corpus)
    actual_corpus_sha256 = hash_ticket35_development_corpus(cases)
    if actual_corpus_sha256 != CORPUS_SHA256:
        raise ValueError(
            "Ticket 35 sparse localization corpus SHA-256 mismatch: "
            f"expected={CORPUS_SHA256} actual={actual_corpus_sha256}"
        )
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(
            "Ticket 35 sparse localization case coverage mismatch: "
            f"expected={EXPECTED_CASE_COUNT} actual={len(cases)}"
        )
    if bundle.schema != BUNDLE_SCHEMA:
        raise ValueError(f"Ticket 35 sparse localization schema drift: {bundle.schema!r}")
    if bundle.corpus_sha256 != CORPUS_SHA256:
        raise ValueError(
            "Ticket 35 sparse localization bundle corpus SHA-256 mismatch: "
            f"expected={CORPUS_SHA256} actual={bundle.corpus_sha256}"
        )
    expected_rows = tuple(
        _row_from_case(case, instance) for case in cases for instance in case.instances
    )
    try:
        rows = tuple(bundle.instances)
    except TypeError as error:
        raise ValueError(
            "Ticket 35 sparse localization instances malformed: bundle.instances is not iterable"
        ) from error
    _validate_duplicate_ids(rows)
    expected_ids = tuple(row.instance_id for row in expected_rows)
    actual_ids = tuple(
        _require_text(row.instance_id, "instance_id", index)
        for index, row in enumerate(rows)
    )
    if len(rows) != EXPECTED_INSTANCE_COUNT:
        missing = tuple(value for value in expected_ids if value not in set(actual_ids))
        extra = tuple(value for value in actual_ids if value not in set(expected_ids))
        raise ValueError(
            "Ticket 35 sparse localization instance coverage mismatch: "
            f"expected={EXPECTED_INSTANCE_COUNT} actual={len(rows)} "
            f"missing={missing[:3]!r} extra={extra[:3]!r}"
        )
    if actual_ids != expected_ids:
        missing = tuple(value for value in expected_ids if value not in set(actual_ids))
        extra = tuple(value for value in actual_ids if value not in set(expected_ids))
        raise ValueError(
            "Ticket 35 sparse localization instance coverage drift: "
            f"missing={missing[:3]!r} extra={extra[:3]!r}"
        )
    for index, (actual, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        context = f"instance_id={expected.instance_id} case_id={expected.case_id} index={index}"
        if not isinstance(actual, SparseLocalizationInstance):
            raise ValueError(f"Ticket 35 sparse localization malformed row: {context}")
        _validate_row_shape(actual, context)
        _validate_points(actual.points, context)
        for field in ("case_id", "filename", "split", "provenance"):
            value = getattr(actual, field)
            expected_value = getattr(expected, field)
            if value != expected_value:
                raise ValueError(
                    f"Ticket 35 sparse localization {field} mismatch: {context} "
                    f"expected={expected_value!r} actual={value!r}"
                )
        if (actual.class_code, actual.kind) != (expected.class_code, expected.kind):
            raise ValueError(
                f"Ticket 35 sparse localization class/kind mismatch: {context} "
                f"expected={(expected.class_code, expected.kind)!r} "
                f"actual={(actual.class_code, actual.kind)!r}"
            )
        if actual.points != expected.points:
            raise ValueError(
                f"Ticket 35 sparse localization points mismatch: {context} "
                f"expected={expected.points!r} actual={actual.points!r}"
            )
        if any(
            token in value
            for value in (actual.instance_id, actual.case_id, actual.filename)
            for token in (*FINAL_MEMBERS, *RESERVED_FINAL_MEMBER_IDS)
        ):
            raise ValueError(f"Ticket 35 sparse localization reserved final member: {context}")
    _validate_split_disjointness(rows)


def canonical_ticket35_sparse_localization_json(
    bundle: SparseLocalizationBundle,
) -> str:
    """Return stable full-bundle JSON for hashing or the 35.04 input seam."""

    return _canonical_json(_bundle_payload(bundle))


def hash_ticket35_sparse_localization_bundle(bundle: SparseLocalizationBundle) -> str:
    """Return the SHA-256 of canonical full sparse truth."""

    return hashlib.sha256(
        canonical_ticket35_sparse_localization_json(bundle).encode("utf-8")
    ).hexdigest()


def resolve_ticket35_sparse_localization_bundle(
    serialized: str,
    expected_sha256: str,
    corpus: Sequence[DevelopmentCase] | None = None,
) -> SparseLocalizationBundle:
    """Parse and validate a full sidecar, failing closed on schema drift."""

    if not isinstance(serialized, str):
        raise ValueError(
            "Ticket 35 sparse localization bundle must be serialized text: "
            f"actual={type(serialized).__name__}"
        )
    actual_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Ticket 35 sparse localization bundle SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    try:
        payload = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Ticket 35 sparse localization bundle: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("invalid Ticket 35 sparse localization bundle: root must be an object")
    required = {"schema", "corpus_sha256", "instances"}
    if set(payload) != required:
        raise ValueError(
            "Ticket 35 sparse localization bundle fields drift: "
            f"expected={sorted(required)!r} actual={sorted(payload)!r}"
        )
    if payload.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"Ticket 35 sparse localization schema drift: {payload.get('schema')!r}")
    instances = payload.get("instances")
    if not isinstance(instances, list):
        raise ValueError("Ticket 35 sparse localization instances must be a list")
    rows: list[SparseLocalizationInstance] = []
    row_fields = {
        "instance_id",
        "case_id",
        "filename",
        "split",
        "class_code",
        "kind",
        "points",
        "provenance",
    }
    for index, value in enumerate(instances):
        context = f"row_index={index}"
        if not isinstance(value, Mapping):
            raise ValueError(f"Ticket 35 sparse localization malformed row: {context}")
        if set(value) != row_fields:
            raise ValueError(
                f"Ticket 35 sparse localization row fields drift: {context} "
                f"expected={sorted(row_fields)!r} actual={sorted(value)!r}"
            )
        points = value.get("points")
        if not isinstance(points, list):
            raise ValueError(f"Ticket 35 sparse localization points must be a list: {context}")
        parsed_points: list[Point] = []
        for point_index, point in enumerate(points):
            point_context = f"{context} point_index={point_index}"
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(not isinstance(coordinate, int) or isinstance(coordinate, bool) for coordinate in point)
            ):
                raise ValueError(f"Ticket 35 sparse localization malformed point: {point_context}")
            parsed_points.append((point[0], point[1]))
        rows.append(
            SparseLocalizationInstance(
                _text_field(value, "instance_id", context),
                _text_field(value, "case_id", context),
                _text_field(value, "filename", context),
                _text_field(value, "split", context),
                _text_field(value, "class_code", context),
                _text_field(value, "kind", context),
                tuple(parsed_points),
                _text_field(value, "provenance", context),
            )
        )
    bundle = SparseLocalizationBundle(
        payload.get("schema"),
        _text_field(payload, "corpus_sha256", "root"),
        tuple(rows),
    )
    validate_ticket35_sparse_localization_bundle(bundle, corpus)
    if hash_ticket35_sparse_localization_bundle(bundle) != expected_sha256:
        raise ValueError(
            "Ticket 35 sparse localization bundle canonical SHA-256 mismatch: "
            f"expected={expected_sha256} actual={hash_ticket35_sparse_localization_bundle(bundle)}"
        )
    return bundle


def build_ticket35_sparse_localization_manifest(
    bundle: SparseLocalizationBundle | None = None,
) -> dict[str, object]:
    """Build the small tracked manifest; full rows remain available by seam."""

    resolved = build_ticket35_sparse_localization_bundle() if bundle is None else bundle
    validate_ticket35_sparse_localization_bundle(resolved)
    return {
        "schema": MANIFEST_SCHEMA,
        "corpus_sha256": resolved.corpus_sha256,
        "bundle_sha256": hash_ticket35_sparse_localization_bundle(resolved),
        "case_count": EXPECTED_CASE_COUNT,
        "instance_count": len(resolved.instances),
        "splits": list(SPLITS),
        "class_codes": list(CLASS_CODES),
        "kind_by_class": dict(KIND_BY_CLASS),
        "provenance": PROVENANCE,
        "declared_consumer": "ticket35.04_only",
        "truth_form": "sparse_point_scribble",
    }


def canonical_ticket35_sparse_localization_manifest_json(
    manifest: Mapping[str, object],
) -> str:
    return _canonical_json(manifest)


def verify_ticket35_sparse_localization_manifest(
    manifest: Mapping[str, object],
    bundle: SparseLocalizationBundle | None = None,
) -> None:
    resolved = build_ticket35_sparse_localization_bundle() if bundle is None else bundle
    expected = build_ticket35_sparse_localization_manifest(resolved)
    if not isinstance(manifest, Mapping) or dict(manifest) != expected:
        raise ValueError(
            "Ticket 35 sparse localization manifest drift: "
            f"expected={expected!r} actual={dict(manifest) if isinstance(manifest, Mapping) else manifest!r}"
        )


def _row_from_case(case: DevelopmentCase, instance) -> SparseLocalizationInstance:
    defect = instance.defect
    if isinstance(defect, Particle):
        kind = "point"
        points = (defect.center,)
    elif isinstance(defect, Scratch):
        kind = "scribble"
        points = defect.points
    else:
        raise ValueError(
            f"Ticket 35 sparse localization unsupported oracle defect: "
            f"case_id={case.case_id} instance_id={instance.instance_id}"
        )
    return SparseLocalizationInstance(
        instance.instance_id,
        case.case_id,
        case.filename,
        case.split,
        defect.class_code,
        kind,
        tuple(points),
        PROVENANCE,
    )


def _bundle_payload(bundle: SparseLocalizationBundle) -> dict[str, object]:
    return {
        "schema": bundle.schema,
        "corpus_sha256": bundle.corpus_sha256,
        "instances": [
            {
                "instance_id": row.instance_id,
                "case_id": row.case_id,
                "filename": row.filename,
                "split": row.split,
                "class_code": row.class_code,
                "kind": row.kind,
                "points": [list(point) for point in row.points],
                "provenance": row.provenance,
            }
            for row in bundle.instances
        ],
    }


def _validate_duplicate_ids(rows: Sequence[SparseLocalizationInstance]) -> None:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        instance_id = _require_text(getattr(row, "instance_id", None), "instance_id", index)
        if instance_id in seen:
            raise ValueError(
                "Ticket 35 sparse localization duplicate instance_id: "
                f"instance_id={instance_id!r} case_id={getattr(row, 'case_id', None)!r} "
                f"index={index}"
            )
        seen.add(instance_id)


def _validate_row_shape(row: SparseLocalizationInstance, context: str) -> None:
    for field in ("instance_id", "case_id", "filename", "split", "class_code", "kind", "provenance"):
        _require_text(getattr(row, field, None), field, context)
    if row.split not in SPLITS:
        raise ValueError(f"Ticket 35 sparse localization split invalid: {context} split={row.split!r}")
    if row.class_code not in CLASS_CODES:
        raise ValueError(f"Ticket 35 sparse localization class invalid: {context} class_code={row.class_code!r}")
    if row.kind != KIND_BY_CLASS[row.class_code]:
        raise ValueError(
            f"Ticket 35 sparse localization class/kind mismatch: {context} "
            f"class_code={row.class_code!r} kind={row.kind!r}"
        )
    if row.provenance != PROVENANCE:
        raise ValueError(
            f"Ticket 35 sparse localization provenance mismatch: {context} "
            f"provenance={row.provenance!r}"
        )


def _validate_points(points: Sequence[Point], context: str) -> None:
    if not points:
        raise ValueError(f"Ticket 35 sparse localization points missing: {context}")
    for point_index, point in enumerate(points):
        if (
            not isinstance(point, tuple)
            or len(point) != 2
            or any(not isinstance(coordinate, int) or isinstance(coordinate, bool) for coordinate in point)
        ):
            raise ValueError(
                f"Ticket 35 sparse localization point malformed: {context} point_index={point_index}"
            )
        x, y = point
        if not (0 <= x < IMAGE_WIDTH and 0 <= y < IMAGE_HEIGHT):
            raise ValueError(
                f"Ticket 35 sparse localization coordinate out of bounds: {context} "
                f"point_index={point_index} point={point!r} image={IMAGE_WIDTH}x{IMAGE_HEIGHT}"
            )


def _validate_split_disjointness(rows: Sequence[SparseLocalizationInstance]) -> None:
    ids_by_split = {split: set() for split in SPLITS}
    for row in rows:
        ids_by_split[row.split].add(row.instance_id)
    overlap = ids_by_split["train"] & ids_by_split["validation"]
    if overlap:
        raise ValueError(
            "Ticket 35 sparse localization train/validation instance overlap: "
            + ", ".join(sorted(overlap)[:3])
        )


def _require_text(value: object, field: str, context: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Ticket 35 sparse localization {field} missing or invalid: context={context!r}"
        )
    return value


def _text_field(payload: Mapping[str, object], field: str, context: object) -> str:
    return _require_text(payload.get(field), field, context)


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


__all__ = [
    "BUNDLE_SCHEMA",
    "CLASS_CODES",
    "CORPUS_SHA256",
    "EXPECTED_CASE_COUNT",
    "EXPECTED_INSTANCE_COUNT",
    "IMAGE_HEIGHT",
    "IMAGE_WIDTH",
    "KIND_BY_CLASS",
    "MANIFEST_PATH",
    "MANIFEST_SCHEMA",
    "PROVENANCE",
    "SPLITS",
    "SparseLocalizationBundle",
    "SparseLocalizationInstance",
    "build_ticket35_sparse_localization_bundle",
    "build_ticket35_sparse_localization_manifest",
    "canonical_ticket35_sparse_localization_json",
    "canonical_ticket35_sparse_localization_manifest_json",
    "hash_ticket35_sparse_localization_bundle",
    "resolve_ticket35_sparse_localization_bundle",
    "validate_ticket35_sparse_localization_bundle",
    "verify_ticket35_sparse_localization_manifest",
]
