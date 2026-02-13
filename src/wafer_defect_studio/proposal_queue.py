"""Pure Review Queue construction and filtering for :class:`DefectProposal` values.

The queue deliberately has no project or SQLite dependency.  A caller may pass
in proposals and either the latest ``ProposalReviewRevision`` values or simple
status values, which keeps this layer useful to both persistence and UI code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from .detection_windows import Rect
from .proposal_generation import DefectProposal
from .proposal_review import ProposalReviewRevision


_REVIEW_STATUSES = ("unreviewed", "accepted", "rejected", "corrected")
_REVIEW_STATUS_SET = frozenset(_REVIEW_STATUSES)
_CONFIDENCE_FIELDS = {
    "mean": "mean_confidence",
    "mean_confidence": "mean_confidence",
    "confidence": "mean_confidence",
    "peak": "peak_confidence",
    "peak_confidence": "peak_confidence",
}
_DISAGREEMENT_KEYS = frozenset(
    {
        "annotation_disagreement",
        "between_run_disagreement",
        "cross_run_disagreement",
        "disagreement",
        "disagreement_flag",
        "disagreement_runs",
        "has_provenance_disagreement",
        "model_annotation_disagreement",
        "model_annotation_mismatch",
        "model_disagreement",
        "model_disagrees",
        "model_mismatch",
        "model_run_disagreement",
        "provenance_disagreement",
        "provenance_disagrees",
        "provenance_mismatch",
        "run_disagreement",
        "run_disagreement_flag",
        "run_mismatch",
        "runs_disagree",
    }
)
_FLAG_CONTAINERS = frozenset(
    {"flags", "provenance_flags", "quality_flags", "review_flags"}
)
_IMAGE_KEYS = (
    "image_asset_id",
    "image_id",
    "source_image_id",
    "wafer_image_id",
)


@dataclass(frozen=True, slots=True, init=False)
class ReviewQueueFilters:
    """Optional predicates used to select Review Queue items.

    ``low_confidence_threshold`` applies to ``mean_confidence`` by default and
    is strict: a proposal is low confidence when
    ``mean_confidence < low_confidence_threshold``.  Set ``confidence_field``
    to ``"peak_confidence"`` (or ``"peak"``) when peak confidence is desired.

    Boolean predicates are tri-state: ``None`` disables that predicate, while
    ``True`` selects matching items and ``False`` selects non-matching items.
    Common short aliases are accepted to keep call sites readable.
    """

    low_confidence_threshold: float | None
    confidence_field: str
    cross_class_conflict: bool | None
    provenance_disagreement: bool | None
    statuses: tuple[str, ...] | None

    def __init__(
        self,
        low_confidence_threshold: Real | None = None,
        *,
        confidence_field: str = "mean_confidence",
        cross_class_conflict: bool | None = None,
        provenance_disagreement: bool | None = None,
        statuses: str | Sequence[str] | None = None,
        # Short/common aliases.  They are normalized into the explicit fields
        # above; no inference is made from arbitrary provenance values.
        low_confidence_below: Real | None = None,
        low_confidence: Real | None = None,
        confidence_threshold: Real | None = None,
        confidence: str | None = None,
        conflict: bool | None = None,
        class_conflict: bool | None = None,
        conflict_only: bool | None = None,
        disagreement: bool | None = None,
        run_disagreement: bool | None = None,
        disagreement_only: bool | None = None,
        status: str | None = None,
        current_status: str | None = None,
        current_statuses: str | Sequence[str] | None = None,
    ) -> None:
        threshold = _coalesce_alias(
            low_confidence_threshold,
            (low_confidence_below, low_confidence, confidence_threshold),
            "low_confidence_threshold",
        )
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, Real):
                raise TypeError("low_confidence_threshold must be a real number")
            threshold = float(threshold)
            if not isfinite(threshold):
                raise ValueError("low_confidence_threshold must be finite")

        chosen_confidence_field = confidence if confidence is not None else confidence_field
        if not isinstance(chosen_confidence_field, str) or not chosen_confidence_field.strip():
            raise TypeError("confidence_field must be a non-empty string")
        chosen_confidence_field = _normalize_confidence_field(chosen_confidence_field)

        conflict_value = _coalesce_alias(
            cross_class_conflict,
            (conflict, class_conflict, conflict_only),
            "cross_class_conflict",
        )
        disagreement_value = _coalesce_alias(
            provenance_disagreement,
            (disagreement, run_disagreement, disagreement_only),
            "provenance_disagreement",
        )
        for value, name in (
            (conflict_value, "cross_class_conflict"),
            (disagreement_value, "provenance_disagreement"),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be True, False, or None")

        chosen_statuses = _coalesce_alias(
            statuses,
            (status, current_status, current_statuses),
            "statuses",
        )
        normalized_statuses = _normalize_statuses(chosen_statuses)

        object.__setattr__(self, "low_confidence_threshold", threshold)
        object.__setattr__(self, "confidence_field", chosen_confidence_field)
        object.__setattr__(self, "cross_class_conflict", conflict_value)
        object.__setattr__(self, "provenance_disagreement", disagreement_value)
        object.__setattr__(self, "statuses", normalized_statuses)

    # Readable aliases for callers that use the shorter filter vocabulary.
    @property
    def low_confidence(self) -> float | None:
        return self.low_confidence_threshold

    @property
    def low_confidence_below(self) -> float | None:
        return self.low_confidence_threshold

    @property
    def confidence_threshold(self) -> float | None:
        return self.low_confidence_threshold

    @property
    def conflict(self) -> bool | None:
        return self.cross_class_conflict

    @property
    def disagreement(self) -> bool | None:
        return self.provenance_disagreement

    @property
    def status(self) -> tuple[str, ...] | None:
        return self.statuses


@dataclass(frozen=True, slots=True)
class ReviewQueueItem:
    """One immutable queue view over a proposal and its current review state."""

    proposal: DefectProposal
    status: str
    source_rect: Rect
    revision: ProposalReviewRevision | Any | None = None
    cross_class_conflict: bool = False
    provenance_disagreement: bool = False

    @property
    def proposal_id(self) -> str:
        return self.proposal.proposal_id

    @property
    def class_name(self) -> str:
        return self.proposal.class_name

    @property
    def current_status(self) -> str:
        return self.status

    @property
    def current_revision(self) -> ProposalReviewRevision | Any | None:
        return self.revision

    @property
    def latest_revision(self) -> ProposalReviewRevision | Any | None:
        return self.revision

    @property
    def geometry(self) -> Rect:
        return self.source_rect

    @property
    def rect(self) -> Rect:
        return self.source_rect

    @property
    def source_image_rect(self) -> Rect:
        return self.source_rect

    @property
    def area(self) -> int:
        return self.proposal.area

    @property
    def peak_confidence(self) -> float:
        return self.proposal.peak_confidence

    @property
    def mean_confidence(self) -> float:
        return self.proposal.mean_confidence

    @property
    def confidence(self) -> float:
        """The default queue confidence, explicitly the proposal mean."""

        return self.proposal.mean_confidence

    @property
    def has_conflict(self) -> bool:
        return self.cross_class_conflict

    @property
    def conflict(self) -> bool:
        return self.cross_class_conflict

    @property
    def class_conflict(self) -> bool:
        return self.cross_class_conflict

    @property
    def overlap_conflict(self) -> bool:
        return self.cross_class_conflict

    @property
    def is_conflict(self) -> bool:
        return self.cross_class_conflict

    @property
    def has_provenance_disagreement(self) -> bool:
        return self.provenance_disagreement

    @property
    def disagreement(self) -> bool:
        return self.provenance_disagreement

    @property
    def run_disagreement(self) -> bool:
        return self.provenance_disagreement

    @property
    def revision_number(self) -> int:
        if self.revision is None:
            return 0
        value = getattr(self.revision, "revision_number", getattr(self.revision, "revision", 0))
        return int(value) if isinstance(value, Real) and not isinstance(value, bool) else 0


@dataclass(frozen=True, slots=True)
class _RevisionState:
    status: str
    source_rect: Rect | None
    provenance: Mapping[str, Any]
    revision_number: int
    original: ProposalReviewRevision | Any | None


def build_review_queue(
    proposals: Iterable[DefectProposal],
    revisions: Mapping[str, Any] | Iterable[Any] | ReviewQueueFilters | None = None,
    *,
    latest_revisions: Mapping[str, Any] | Iterable[Any] | None = None,
    review_revisions: Mapping[str, Any] | Iterable[Any] | None = None,
    latest_statuses: Mapping[str, str] | None = None,
    filters: ReviewQueueFilters | Mapping[str, Any] | None = None,
    **filter_options: Any,
) -> tuple[ReviewQueueItem, ...]:
    """Build a deterministic pure queue from proposals and optional reviews.

    Proposals are sorted by immutable ``proposal_id``.  Revisions may be a
    mapping from proposal ID to a latest revision/status, or an iterable of
    revisions; when multiple revisions are supplied, the greatest revision
    number is selected.  No database is opened or written.
    """

    if isinstance(revisions, ReviewQueueFilters):
        if filters is not None:
            raise TypeError("filters were supplied twice")
        filters = revisions
        revisions = None
    chosen_revisions = _coalesce_revision_input(revisions, latest_revisions, review_revisions)
    status_overrides = dict(latest_statuses or {})
    proposal_values = tuple(proposals)
    _validate_proposals(proposal_values)
    states = _latest_revision_states(chosen_revisions)

    provisional: list[ReviewQueueItem] = []
    for proposal in sorted(proposal_values, key=lambda value: str(value.proposal_id)):
        state = states.get(proposal.proposal_id)
        if proposal.proposal_id in status_overrides:
            state = _RevisionState(
                status=_normalize_status(status_overrides[proposal.proposal_id]),
                source_rect=state.source_rect if state is not None else None,
                provenance=state.provenance if state is not None else {},
                revision_number=state.revision_number if state is not None else 0,
                original=state.original if state is not None else None,
            )
        status = state.status if state is not None else "unreviewed"
        rect = state.source_rect if state is not None and state.source_rect is not None else proposal.source_rect
        disagreement = _has_provenance_disagreement(
            proposal.provenance,
            state.provenance if state is not None else None,
        )
        provisional.append(
            ReviewQueueItem(
                proposal=proposal,
                status=status,
                source_rect=rect,
                revision=state.original if state is not None else None,
                provenance_disagreement=disagreement,
            )
        )

    conflict_indices: set[int] = set()
    for index, left in enumerate(provisional):
        for other_index in range(index + 1, len(provisional)):
            right = provisional[other_index]
            if left.class_name == right.class_name:
                continue
            if not _same_source_image(left.proposal, right.proposal):
                continue
            if _rects_overlap(left.source_rect, right.source_rect):
                conflict_indices.update((index, other_index))

    queue = tuple(
        ReviewQueueItem(
            proposal=item.proposal,
            status=item.status,
            source_rect=item.source_rect,
            revision=item.revision,
            cross_class_conflict=index in conflict_indices,
            provenance_disagreement=item.provenance_disagreement,
        )
        for index, item in enumerate(provisional)
    )
    chosen_filters = _coerce_filters(filters, filter_options)
    return _apply_filters(queue, chosen_filters)


def filter_review_queue(
    items_or_proposals: Iterable[ReviewQueueItem] | Iterable[DefectProposal],
    revisions: Mapping[str, Any] | Iterable[Any] | ReviewQueueFilters | None = None,
    filters: ReviewQueueFilters | Mapping[str, Any] | None = None,
    **filter_options: Any,
) -> tuple[ReviewQueueItem, ...]:
    """Apply pure filters to queue items or build a queue from proposals.

    Passing already-built ``ReviewQueueItem`` values avoids recomputing flags;
    passing ``DefectProposal`` values delegates to :func:`build_review_queue`.
    """

    if isinstance(revisions, ReviewQueueFilters):
        if filters is not None:
            raise TypeError("filters were supplied twice")
        filters = revisions
        revisions = None
    values = tuple(items_or_proposals)
    if values and all(isinstance(value, ReviewQueueItem) for value in values):
        queue = tuple(sorted(values, key=lambda value: str(value.proposal_id)))
    elif not values:
        queue = ()
    else:
        queue = build_review_queue(values, revisions)
    chosen_filters = _coerce_filters(filters, filter_options)
    return _apply_filters(queue, chosen_filters)


def _coalesce_revision_input(
    revisions: Mapping[str, Any] | Iterable[Any] | None,
    latest_revisions: Mapping[str, Any] | Iterable[Any] | None,
    review_revisions: Mapping[str, Any] | Iterable[Any] | None,
) -> Mapping[str, Any] | Iterable[Any] | None:
    chosen = revisions
    for value, name in ((latest_revisions, "latest_revisions"), (review_revisions, "review_revisions")):
        if value is not None:
            if chosen is not None:
                raise TypeError(f"{name} was supplied alongside revisions")
            chosen = value
    return chosen


def _validate_proposals(proposals: Sequence[DefectProposal]) -> None:
    seen: set[str] = set()
    for proposal in proposals:
        if not isinstance(proposal, DefectProposal):
            raise TypeError("proposals must contain DefectProposal values")
        proposal_id = str(proposal.proposal_id)
        if not proposal_id:
            raise ValueError("proposal_id must be non-empty")
        if proposal_id in seen:
            raise ValueError(f"duplicate proposal_id: {proposal_id}")
        seen.add(proposal_id)


def _latest_revision_states(
    revisions: Mapping[str, Any] | Iterable[Any] | None,
) -> dict[str, _RevisionState]:
    if revisions is None:
        return {}
    states: dict[str, _RevisionState] = {}
    if isinstance(revisions, Mapping):
        for proposal_id, value in revisions.items():
            identifier = str(proposal_id)
            candidate = _latest_state(value, identifier)
            if candidate is not None:
                states[identifier] = candidate
        return states
    try:
        values = tuple(revisions)
    except TypeError as error:
        raise TypeError("revisions must be a mapping or iterable") from error
    candidates: dict[str, list[_RevisionState]] = {}
    for value in values:
        state = _state_from_value(value)
        identifier = _revision_proposal_id(value)
        if state is None or identifier is None:
            raise ValueError("each revision must identify a proposal and status")
        candidates.setdefault(identifier, []).append(state)
    for identifier, options in candidates.items():
        states[identifier] = max(options, key=lambda value: value.revision_number)
    return states


def _latest_state(value: Any, proposal_id: str) -> _RevisionState | None:
    if value is None:
        return None
    if _is_revision_sequence(value):
        options = tuple(_state_from_value(item) for item in value)
        chosen = tuple(option for option in options if option is not None)
        if not chosen:
            return None
        return max(chosen, key=lambda item: item.revision_number)
    state = _state_from_value(value)
    if state is None:
        # A mapping from ID to a bare status is intentionally accepted.
        if isinstance(value, (str,)):
            return _RevisionState(_normalize_status(value), None, {}, 0, None)
        raise ValueError(f"invalid review revision for proposal {proposal_id}")
    return state


def _is_revision_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, Mapping, ProposalReviewRevision)):
        return False
    return isinstance(value, Sequence) or isinstance(value, Iterable)


def _state_from_value(value: Any) -> _RevisionState | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _RevisionState(_normalize_status(value), None, {}, 0, None)
    if isinstance(value, Mapping):
        if "status" not in value:
            return None
        status = _normalize_status(value["status"])
        source_rect = value.get("source_rect", value.get("rect", value.get("geometry")))
        rect = _coerce_rect(source_rect) if source_rect is not None else None
        provenance = value.get("provenance", {})
        if not isinstance(provenance, Mapping):
            raise TypeError("review provenance must be a mapping")
        revision_number = value.get("revision_number", value.get("revision", 0))
        return _RevisionState(
            status,
            rect,
            dict(provenance),
            _revision_number(revision_number),
            value,
        )
    status = getattr(value, "status", None)
    if status is None:
        return None
    source_rect = getattr(value, "source_rect", getattr(value, "rect", getattr(value, "geometry", None)))
    rect = _coerce_rect(source_rect) if source_rect is not None else None
    provenance = getattr(value, "provenance", {})
    if not isinstance(provenance, Mapping):
        raise TypeError("review provenance must be a mapping")
    revision_number = getattr(value, "revision_number", getattr(value, "revision", 0))
    return _RevisionState(
        _normalize_status(status),
        rect,
        dict(provenance),
        _revision_number(revision_number),
        value,
    )


def _revision_proposal_id(value: Any) -> str | None:
    if isinstance(value, Mapping):
        identifier = value.get("proposal_id", value.get("id"))
    else:
        identifier = getattr(value, "proposal_id", getattr(value, "id", None))
    if identifier is None:
        return None
    return str(identifier)


def _revision_number(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        return 0
    return int(value)


def _coerce_rect(value: Any) -> Rect:
    if isinstance(value, Rect):
        return value
    if isinstance(value, Mapping):
        values = (value.get("x"), value.get("y"), value.get("width"), value.get("height"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 4:
        values = tuple(value)
    else:
        raise TypeError("review source_rect must be a Rect, mapping, or four-value sequence")
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in values):
        raise TypeError("review source_rect coordinates must be numeric")
    return Rect(*(int(item) for item in values))


def _normalize_status(value: Any) -> str:
    if hasattr(value, "value") and isinstance(value.value, str):
        value = value.value
    if not isinstance(value, str):
        raise TypeError("review status must be a string")
    status = value.strip().lower()
    if status not in _REVIEW_STATUS_SET:
        raise ValueError(f"unsupported proposal review status: {value!r}")
    return status


def _normalize_statuses(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence):
        values = tuple(value)
    else:
        raise TypeError("statuses must be a status string or sequence")
    normalized = tuple(sorted({_normalize_status(item) for item in values}))
    return normalized


def _normalize_confidence_field(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return _CONFIDENCE_FIELDS.get(normalized, normalized)


def _coalesce_alias(primary: Any, aliases: Sequence[Any], name: str) -> Any:
    chosen = primary
    for value in aliases:
        if value is None:
            continue
        if chosen is not None and value != chosen:
            raise ValueError(f"conflicting values for {name}")
        chosen = value
    return chosen


def _coerce_filters(
    filters: ReviewQueueFilters | Mapping[str, Any] | None,
    options: Mapping[str, Any],
) -> ReviewQueueFilters:
    if filters is None:
        return ReviewQueueFilters(**dict(options))
    if options:
        raise TypeError("filter options were supplied alongside filters")
    if isinstance(filters, ReviewQueueFilters):
        return filters
    if isinstance(filters, Mapping):
        return ReviewQueueFilters(**dict(filters))
    raise TypeError("filters must be ReviewQueueFilters or a mapping")


def _apply_filters(
    queue: Sequence[ReviewQueueItem],
    filters: ReviewQueueFilters,
) -> tuple[ReviewQueueItem, ...]:
    result: list[ReviewQueueItem] = []
    for item in queue:
        if filters.low_confidence_threshold is not None:
            confidence = _item_confidence(item, filters.confidence_field)
            if not confidence < filters.low_confidence_threshold:
                continue
        if (
            filters.cross_class_conflict is not None
            and item.cross_class_conflict != filters.cross_class_conflict
        ):
            continue
        if (
            filters.provenance_disagreement is not None
            and item.provenance_disagreement != filters.provenance_disagreement
        ):
            continue
        if filters.statuses is not None and item.status not in filters.statuses:
            continue
        result.append(item)
    return tuple(result)


def _item_confidence(item: ReviewQueueItem, field: str) -> float:
    normalized = _normalize_confidence_field(field)
    if normalized in {"mean_confidence", "peak_confidence"}:
        value = getattr(item, normalized)
    else:
        value = getattr(item, normalized, getattr(item.proposal, normalized, None))
        if value is None and isinstance(item.proposal.provenance, Mapping):
            value = item.proposal.provenance.get(normalized)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"confidence field is not numeric: {field!r}")
    return float(value)


def _has_provenance_disagreement(
    *provenances: Mapping[str, Any] | None,
) -> bool:
    for provenance in provenances:
        if not isinstance(provenance, Mapping):
            continue
        if _has_explicit_flag(provenance):
            return True
    return False


def _has_explicit_flag(provenance: Mapping[str, Any]) -> bool:
    for key, value in provenance.items():
        normalized = _normalize_key(key)
        if normalized in _DISAGREEMENT_KEYS:
            if isinstance(value, bool) and value:
                return True
            if normalized == "disagreement_runs" and _nonempty(value):
                return True
        if normalized in _FLAG_CONTAINERS and isinstance(value, Mapping) and _has_explicit_flag(value):
            return True
    return False


def _nonempty(value: Any) -> bool:
    """Return true for a non-empty disagreement-runs collection."""

    if isinstance(value, (str, bytes, bytearray)):
        return bool(value)
    try:
        return len(value) > 0
    except TypeError:
        return False


def _normalize_key(value: Any) -> str:
    return "_".join("".join(character.lower() if character.isalnum() else "_" for character in str(value)).split("_"))


def _same_source_image(left: DefectProposal, right: DefectProposal) -> bool:
    left_id = _image_identity(left.provenance)
    right_id = _image_identity(right.provenance)
    return left_id is None or right_id is None or left_id == right_id


def _image_identity(provenance: Mapping[str, Any]) -> str | None:
    for key in _IMAGE_KEYS:
        value = provenance.get(key)
        if value is not None and str(value):
            return str(value)
    return None


def _rects_overlap(left: Rect, right: Rect) -> bool:
    return (
        max(left.x, right.x) < min(left.right, right.right)
        and max(left.y, right.y) < min(left.bottom, right.bottom)
    )


build_queue = build_review_queue
filter_queue = filter_review_queue


__all__ = [
    "ReviewQueueFilters",
    "ReviewQueueItem",
    "build_review_queue",
    "filter_review_queue",
    "build_queue",
    "filter_queue",
]
