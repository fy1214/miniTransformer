"""Token/expert alignment between the two frameworks' capture records.

Given the records of one run directory for framework A (e.g. sglang) and
framework B (e.g. megatron), pair records that describe the same logical
token/expert tensor so downstream metrics compare like with like.

Alignment key: ``(sample_id, layer_id, expert_id, token_position,
route_slot)``.  Records are aligned independently per ``expert_id`` group
(records with ``expert_id=None`` form their own group) and per
``tensor_point``: two records that share the alignment key but differ in
``tensor_point`` are different capture points of the same token and are
never paired with each other.

Method priority (fixed, highest first):

1. ``dispatcher_index`` -- when the dispatcher's token/permutation index
   was captured, records with non-None ``token_position`` are paired by
   exact key match.  Payloads are NOT loaded in this stage: dispatcher
   indices are trusted.  A pair whose shapes contradict each other is
   rejected as a *conflict* and invalidates the expert.
2. ``hash_exact`` -- still-unpaired records where at least one side lacks
   a dispatcher index are matched on payload bytes: a SHA-256 over
   ``(dtype, shape, stride, contiguous payload bytes)`` acts as a coarse
   filter, then ``torch.equal`` verifies the candidate.  Matching infers
   the missing ``token_position`` (from the keyed side when exactly one
   side has one, else by enumeration inside the match bucket).  Payloads
   are loaded only from this stage on, always via
   :func:`logpdiff.writer.assert_internal_payload` resolved against the
   run directory of their own side.

   Caveat: ``torch.equal`` treats ``NaN != NaN``, so byte-identical
   payloads containing NaN do not verify; such records stay unmatched.
3. ``cosine_fallback`` -- only when ``allow_value_match_fallback=True``:
   remaining records are paired greedily by payload cosine similarity
   (threshold ``COSINE_FALLBACK_THRESHOLD``).  The minimum accepted
   cosine is reported in ``AlignmentResult.min_fallback_cosine``.

Two records that BOTH carry dispatcher indices are never value-matched
against each other: if their keys differ, the dispatcher says they are
different tokens, and pairing them by value would hide routing bugs.

Semantics:

- *duplicate*: same side, same full key (only keys with non-None
  ``token_position`` can collide).  The first record stays the match
  candidate; later copies are counted and reported unmatched.
- *conflict*: a dispatcher-key pair with contradictory shapes
  (unexplained -> the expert is invalidated), or a payload that verifies
  against more than one candidate on the other side (explained by
  identical token content -> counted, but the expert stays valid).
- ``one_to_one_per_expert[e]`` is True iff expert ``e`` paired up exactly:
  no unmatched records, no duplicates, no conflicts.
- Experts with an unexplained conflict land in ``invalid_experts``; the
  upper layer must mark their numeric metrics ``ALIGNMENT_INVALID``
  instead of reporting a misleading cosine.

Only the standard library is required at import time; torch is imported
lazily when payloads actually need loading.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .schema import TensorRecord
from .writer import assert_internal_payload

ALIGNMENT_INVALID = "INVALID_ALIGNMENT"
"""Metric value the upper layer must emit for experts in ``invalid_experts``."""

METHOD_DISPATCHER_INDEX = "dispatcher_index"
METHOD_HASH_EXACT = "hash_exact"
METHOD_COSINE_FALLBACK = "cosine_fallback"

COSINE_FALLBACK_THRESHOLD = 0.999
"""Minimum cosine similarity accepted by the value-match fallback."""

_METHOD_RANK = {
    METHOD_DISPATCHER_INDEX: 0,
    METHOD_HASH_EXACT: 1,
    METHOD_COSINE_FALLBACK: 2,
}


@dataclass
class AlignedPair:
    """Two records (one per side) believed to be the same logical tensor."""

    sample_id: Any
    layer_id: int
    expert_id: Optional[int]
    token_position: Optional[int]
    route_slot: Optional[int]
    record_a: TensorRecord
    record_b: TensorRecord
    method: str


@dataclass
class AlignmentResult:
    """Outcome of :func:`align_records`.

    ``method`` is the weakest method any matched pair relied on
    (``dispatcher_index`` < ``hash_exact`` < ``cosine_fallback``), so a
    single fallback pair marks the whole result as ``cosine_fallback``.
    ``one_to_one_per_expert`` / ``invalid_experts`` only cover integer
    expert ids; the ``expert_id=None`` group is still aligned and counted
    in ``matched`` / ``unmatched_*`` / ``duplicate_count`` /
    ``conflict_count``.
    """

    method: str
    matched: List[AlignedPair] = field(default_factory=list)
    unmatched_a: List[TensorRecord] = field(default_factory=list)
    unmatched_b: List[TensorRecord] = field(default_factory=list)
    duplicate_count: int = 0
    conflict_count: int = 0
    min_fallback_cosine: Optional[float] = None
    one_to_one_per_expert: Dict[int, bool] = field(default_factory=dict)
    invalid_experts: Set[int] = field(default_factory=set)


# ---------------------------------------------------------------------------
# payload loading (torch imported lazily)
# ---------------------------------------------------------------------------


def _import_torch():
    import torch  # noqa: F401 -- lazy: aligner must import without torch

    return torch


class _PayloadLoader:
    """Loads and caches payload tensors, one run directory per side."""

    def __init__(self, run_dir_a, run_dir_b) -> None:
        self._run_dirs = {"a": run_dir_a, "b": run_dir_b}
        self._torch = None
        self._cache: Dict[Tuple[str, str], Any] = {}

    @property
    def torch(self):
        if self._torch is None:
            self._torch = _import_torch()
        return self._torch

    def load(self, side: str, record: TensorRecord):
        key = (side, record.payload_path)
        if key not in self._cache:
            path = assert_internal_payload(
                record.payload_path, self._run_dirs[side]
            )
            self._cache[key] = self.torch.load(path, map_location="cpu")
        return self._cache[key]


def _payload_hash(t, torch) -> str:
    """SHA-256 over dtype, shape, stride and contiguous payload bytes."""
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("utf-8"))
    h.update(repr(tuple(t.shape)).encode("utf-8"))
    h.update(repr(tuple(t.stride())).encode("utf-8"))
    c = t.detach().cpu().contiguous()
    if c.numel() > 0:
        flat = c.reshape(-1)
        try:
            # reinterpret as uint8 so bf16/fp16 payloads hash by raw bytes
            raw = flat.view(torch.uint8).numpy().tobytes()
        except (TypeError, RuntimeError):
            import io

            buf = io.BytesIO()
            torch.save(c, buf)
            raw = buf.getvalue()
        h.update(raw)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# key helpers
# ---------------------------------------------------------------------------


def _key(r: TensorRecord) -> Tuple:
    """Full identity used for matching; ``tensor_point`` is part of it."""
    return (
        r.sample_id,
        r.layer_id,
        r.tensor_point,
        r.token_position,
        r.route_slot,
    )


def _value_bucket(r: TensorRecord) -> Tuple:
    """Records only value-match within the same sample/layer/capture point."""
    return (r.sample_id, r.layer_id, r.tensor_point)


def _dedupe(records: List[TensorRecord]) -> Tuple[List[TensorRecord], List[TensorRecord]]:
    """Split into first-occurrence representatives and duplicate extras.

    Records with ``token_position=None`` can never be duplicates: without a
    dispatcher index two records of the same capture point may be distinct
    tokens, so each is kept as its own candidate.
    """
    seen: Set[Tuple] = set()
    reps: List[TensorRecord] = []
    extras: List[TensorRecord] = []
    for r in records:
        if r.token_position is None:
            reps.append(r)
            continue
        k = _key(r)
        if k in seen:
            extras.append(r)
        else:
            seen.add(k)
            reps.append(r)
    return reps, extras


def _both_keyed(ra: TensorRecord, rb: TensorRecord) -> bool:
    return ra.token_position is not None and rb.token_position is not None


def _infer_position(
    ra: TensorRecord, rb: TensorRecord, counters: Dict[Tuple, int]
) -> Tuple[Optional[int], Optional[int]]:
    """Token position for a value-matched pair: trust the keyed side, else
    enumerate pairs inside the value bucket in a-side appearance order."""
    tok = ra.token_position if ra.token_position is not None else rb.token_position
    slot = ra.route_slot if ra.route_slot is not None else rb.route_slot
    if tok is None:
        bucket = _value_bucket(ra)
        tok = counters.get(bucket, 0)
        counters[bucket] = tok + 1
    return tok, slot


def _cosine(ta, tb, torch) -> Optional[float]:
    """Cosine similarity of two equal-shape payloads, None when undefined."""
    va = ta.detach().reshape(-1).to(torch.float64)
    vb = tb.detach().reshape(-1).to(torch.float64)
    na = float(va.norm())
    nb = float(vb.norm())
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(va, vb) / (na * nb))


# ---------------------------------------------------------------------------
# per-expert-group alignment
# ---------------------------------------------------------------------------


def _align_group(
    expert_id: Optional[int],
    recs_a: List[TensorRecord],
    recs_b: List[TensorRecord],
    loader: _PayloadLoader,
    allow_fallback: bool,
    result: AlignmentResult,
) -> None:
    one_to_one = True
    unexplained_conflict = False
    pairs: List[AlignedPair] = []

    reps_a, extras_a = _dedupe(recs_a)
    reps_b, extras_b = _dedupe(recs_b)
    if extras_a or extras_b:
        one_to_one = False
    result.duplicate_count += len(extras_a) + len(extras_b)
    unmatched_a = list(extras_a)
    unmatched_b = list(extras_b)

    # -- stage 1: dispatcher index, exact key match, payloads untouched ----
    keyed_b: Dict[Tuple, TensorRecord] = {}
    for r in reps_b:
        if r.token_position is not None:
            keyed_b[_key(r)] = r
    consumed_b: Set[Tuple] = set()
    remaining_a: List[TensorRecord] = []
    for ra in reps_a:
        if ra.token_position is None:
            remaining_a.append(ra)
            continue
        k = _key(ra)
        rb = keyed_b.get(k)
        if rb is None:
            remaining_a.append(ra)
            continue
        consumed_b.add(k)
        if ra.shape and rb.shape and tuple(ra.shape) != tuple(rb.shape):
            # dispatcher says same token but shapes disagree: unexplained
            result.conflict_count += 1
            one_to_one = False
            unexplained_conflict = True
            unmatched_a.append(ra)
            unmatched_b.append(rb)
            continue
        pairs.append(
            AlignedPair(
                sample_id=ra.sample_id,
                layer_id=ra.layer_id,
                expert_id=ra.expert_id,
                token_position=ra.token_position,
                route_slot=ra.route_slot,
                record_a=ra,
                record_b=rb,
                method=METHOD_DISPATCHER_INDEX,
            )
        )
    remaining_b = [
        r
        for r in reps_b
        if r.token_position is None or _key(r) not in consumed_b
    ]

    # -- stage 2: payload byte hash coarse filter + torch.equal verify -----
    pos_counters: Dict[Tuple, int] = {}
    cand_a = [r for r in remaining_a if r.payload_path]
    cand_b = [r for r in remaining_b if r.payload_path]
    if cand_a and cand_b:
        torch = loader.torch
        buckets_b: Dict[Tuple, List[Tuple[TensorRecord, Any]]] = {}
        for rb in cand_b:
            tb = loader.load("b", rb)
            bk = _value_bucket(rb) + (_payload_hash(tb, torch),)
            buckets_b.setdefault(bk, []).append((rb, tb))
        used_b: Set[int] = set()
        still_a: List[TensorRecord] = []
        for ra in cand_a:
            ta = loader.load("a", ra)
            bk = _value_bucket(ra) + (_payload_hash(ta, torch),)
            verified = []
            for rb, tb in buckets_b.get(bk, []):
                if id(rb) in used_b or _both_keyed(ra, rb):
                    continue
                if torch.equal(ta, tb):
                    verified.append(rb)
            if not verified:
                still_a.append(ra)
                continue
            if len(verified) > 1:
                # byte-identical duplicates on the other side: ambiguous but
                # explained, so counted without invalidating the expert
                result.conflict_count += 1
                one_to_one = False
            rb = verified[0]
            used_b.add(id(rb))
            tok, slot = _infer_position(ra, rb, pos_counters)
            pairs.append(
                AlignedPair(
                    sample_id=ra.sample_id,
                    layer_id=ra.layer_id,
                    expert_id=ra.expert_id,
                    token_position=tok,
                    route_slot=slot,
                    record_a=ra,
                    record_b=rb,
                    method=METHOD_HASH_EXACT,
                )
            )
        remaining_a = still_a + [r for r in remaining_a if not r.payload_path]
        remaining_b = [r for r in remaining_b if not r.payload_path] + [
            rb for rb in cand_b if id(rb) not in used_b
        ]

    # -- stage 3: cosine fallback (opt-in) ----------------------------------
    if allow_fallback:
        cand_a = [r for r in remaining_a if r.payload_path]
        cand_b = [r for r in remaining_b if r.payload_path]
        if cand_a and cand_b:
            torch = loader.torch
            scored = []
            for i, ra in enumerate(cand_a):
                ta = loader.load("a", ra)
                for j, rb in enumerate(cand_b):
                    if _value_bucket(ra) != _value_bucket(rb):
                        continue
                    if _both_keyed(ra, rb):
                        continue
                    tb = loader.load("b", rb)
                    if tuple(ta.shape) != tuple(tb.shape):
                        continue
                    c = _cosine(ta, tb, torch)
                    if c is not None and c >= COSINE_FALLBACK_THRESHOLD:
                        # sort by cosine desc, then a/b order for determinism
                        scored.append((-c, i, j, c))
            scored.sort()
            used_ai: Set[int] = set()
            used_bi: Set[int] = set()
            for _negc, i, j, c in scored:
                if i in used_ai or j in used_bi:
                    continue
                used_ai.add(i)
                used_bi.add(j)
                ra, rb = cand_a[i], cand_b[j]
                tok, slot = _infer_position(ra, rb, pos_counters)
                pairs.append(
                    AlignedPair(
                        sample_id=ra.sample_id,
                        layer_id=ra.layer_id,
                        expert_id=ra.expert_id,
                        token_position=tok,
                        route_slot=slot,
                        record_a=ra,
                        record_b=rb,
                        method=METHOD_COSINE_FALLBACK,
                    )
                )
                if result.min_fallback_cosine is None:
                    result.min_fallback_cosine = c
                else:
                    result.min_fallback_cosine = min(result.min_fallback_cosine, c)
            remaining_a = [r for i, r in enumerate(cand_a) if i not in used_ai] + [
                r for r in remaining_a if not r.payload_path
            ]
            remaining_b = [r for j, r in enumerate(cand_b) if j not in used_bi] + [
                r for r in remaining_b if not r.payload_path
            ]

    unmatched_a.extend(remaining_a)
    unmatched_b.extend(remaining_b)
    if remaining_a or remaining_b:
        one_to_one = False

    result.matched.extend(pairs)
    result.unmatched_a.extend(unmatched_a)
    result.unmatched_b.extend(unmatched_b)
    if expert_id is not None:
        result.one_to_one_per_expert[expert_id] = one_to_one
        if unexplained_conflict:
            result.invalid_experts.add(expert_id)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def align_records(
    records_a: List[TensorRecord],
    records_b: List[TensorRecord],
    run_dir_a,
    run_dir_b,
    *,
    allow_value_match_fallback: bool = False,
) -> AlignmentResult:
    """Align two frameworks' records per expert group.

    ``records_a`` / ``records_b`` are :class:`TensorRecord` lists from the
    two sides' ``index.jsonl`` files; ``run_dir_a`` / ``run_dir_b`` are the
    run directories payloads resolve against.  Payload files are only read
    when dispatcher indices are insufficient (hash stage, or the opt-in
    cosine fallback), and only via
    :func:`logpdiff.writer.assert_internal_payload`.
    """
    groups_a: Dict[Optional[int], List[TensorRecord]] = {}
    groups_b: Dict[Optional[int], List[TensorRecord]] = {}
    for r in records_a:
        groups_a.setdefault(r.expert_id, []).append(r)
    for r in records_b:
        groups_b.setdefault(r.expert_id, []).append(r)
    expert_ids = sorted(
        set(groups_a) | set(groups_b), key=lambda e: (e is None, e)
    )

    result = AlignmentResult(method=METHOD_DISPATCHER_INDEX)
    loader = _PayloadLoader(run_dir_a, run_dir_b)
    for expert_id in expert_ids:
        _align_group(
            expert_id,
            groups_a.get(expert_id, []),
            groups_b.get(expert_id, []),
            loader,
            allow_value_match_fallback,
            result,
        )
    if result.matched:
        result.method = max(
            (p.method for p in result.matched), key=_METHOD_RANK.__getitem__
        )
    return result
