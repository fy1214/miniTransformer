"""Tests for logpdiff.aligner: dispatcher keys, hash matching, fallback."""

from __future__ import annotations

import pytest

from logpdiff import schema
from logpdiff.aligner import (
    COSINE_FALLBACK_THRESHOLD,
    METHOD_COSINE_FALLBACK,
    METHOD_DISPATCHER_INDEX,
    METHOD_HASH_EXACT,
    align_records,
)
from logpdiff.writer import create_run_dir

from conftest import make_record, require_torch, save_payload


@pytest.fixture
def run(tmp_path):
    return create_run_dir(tmp_path, run_id="align-run")


def _keyed(framework, tok, **kwargs):
    """Record with a dispatcher index; payload_path may point nowhere because
    the dispatcher-index stage must never load payloads."""
    kwargs.setdefault("expert_id", 0)
    kwargs.setdefault("route_slot", 0)
    kwargs.setdefault("payload_path", f"{framework}/tensors/never-written.pt")
    tensor_point = kwargs.pop("tensor_point", schema.FC1_OUTPUT)
    return make_record("align-run", framework, tensor_point,
                       token_position=tok, **kwargs)


class TestDispatcherIndex:
    def test_exact_key_match_without_loading_payloads(self, run):
        recs_a = [_keyed("sglang", t) for t in (0, 1)]
        recs_b = [_keyed("megatron", t) for t in (0, 1)]
        # payload files do not exist: if the dispatcher stage read them,
        # this would raise instead of matching.
        result = align_records(recs_a, recs_b, run.run_dir, run.run_dir)
        assert result.method == METHOD_DISPATCHER_INDEX
        assert len(result.matched) == 2
        assert {p.token_position for p in result.matched} == {0, 1}
        assert result.unmatched_a == [] and result.unmatched_b == []
        assert result.duplicate_count == 0
        assert result.conflict_count == 0
        assert result.one_to_one_per_expert == {0: True}
        assert result.invalid_experts == set()

    def test_tensor_point_is_part_of_the_key(self, run):
        # same token/expert but different capture points never pair up
        torch = require_torch()
        t = torch.randn(8)
        ra = save_payload(run, "sglang", schema.FC1_OUTPUT, t,
                          expert_id=0, token_position=0, route_slot=0)
        rb = save_payload(run, "megatron", schema.FC2_OUTPUT, t,
                          expert_id=0, token_position=0, route_slot=0)
        result = align_records([ra], [rb], run.run_dir, run.run_dir)
        assert result.matched == []
        assert len(result.unmatched_a) == 1
        assert len(result.unmatched_b) == 1

    def test_conflicting_shapes_invalidate_expert(self, run):
        ra = _keyed("sglang", 0, shape=(8,))
        rb = _keyed("megatron", 0, shape=(16,))
        result = align_records([ra], [rb], run.run_dir, run.run_dir)
        assert result.matched == []
        assert result.conflict_count == 1
        assert result.invalid_experts == {0}
        assert result.one_to_one_per_expert[0] is False
        assert len(result.unmatched_a) == 1
        assert len(result.unmatched_b) == 1

    def test_duplicate_records_counted(self, run):
        ra1 = _keyed("sglang", 0)
        ra2 = _keyed("sglang", 0)  # same full key as ra1: duplicate
        rb = _keyed("megatron", 0)
        result = align_records([ra1, ra2], [rb], run.run_dir, run.run_dir)
        assert result.duplicate_count == 1
        assert len(result.matched) == 1
        assert result.matched[0].record_a is ra1  # first occurrence wins
        assert result.unmatched_a == [ra2]
        assert result.one_to_one_per_expert[0] is False

    def test_both_keyed_never_value_matched(self, run):
        """Same payload bytes, different dispatcher keys -> NOT a pair.

        Trusting the dispatcher over values is what keeps routing bugs
        visible; this must hold even with the cosine fallback enabled.
        """
        torch = require_torch()
        t = torch.randn(8)
        ra = save_payload(run, "sglang", schema.FC1_OUTPUT, t,
                          expert_id=0, token_position=0, route_slot=0)
        rb = save_payload(run, "megatron", schema.FC1_OUTPUT, t.clone(),
                          expert_id=0, token_position=1, route_slot=0)
        for allow in (False, True):
            result = align_records(
                [ra], [rb], run.run_dir, run.run_dir,
                allow_value_match_fallback=allow,
            )
            assert result.matched == []
            assert result.unmatched_a == [ra]
            assert result.unmatched_b == [rb]
            assert result.one_to_one_per_expert[0] is False


class TestHashExact:
    def test_bf16_payload_matches_and_infers_position(self, run):
        torch = require_torch()
        t = torch.randn(8, dtype=torch.bfloat16)
        ra = save_payload(run, "sglang", schema.FC1_OUTPUT, t,
                          expert_id=0)  # no dispatcher index
        rb = save_payload(run, "megatron", schema.FC1_OUTPUT, t,
                          expert_id=0, token_position=3, route_slot=1)
        result = align_records([ra], [rb], run.run_dir, run.run_dir)
        assert result.method == METHOD_HASH_EXACT
        assert len(result.matched) == 1
        pair = result.matched[0]
        assert pair.token_position == 3  # inferred from the keyed side
        assert pair.route_slot == 1
        assert pair.record_a is ra and pair.record_b is rb
        assert result.one_to_one_per_expert == {0: True}

    def test_byte_identical_candidates_are_explained_conflicts(self, run):
        torch = require_torch()
        t = torch.randn(8)
        ra = save_payload(run, "sglang", schema.FC1_OUTPUT, t, expert_id=0)
        rb1 = save_payload(run, "megatron", schema.FC1_OUTPUT, t, expert_id=0)
        rb2 = save_payload(run, "megatron", schema.FC1_OUTPUT, t.clone(),
                           expert_id=0)
        result = align_records([ra], [rb1, rb2], run.run_dir, run.run_dir)
        assert len(result.matched) == 1
        assert result.conflict_count == 1  # ambiguous but explained
        assert result.invalid_experts == set()  # expert stays valid
        assert result.one_to_one_per_expert[0] is False
        assert len(result.unmatched_b) == 1

    def test_none_expert_group_aligns(self, run):
        torch = require_torch()
        t = torch.randn(4, 4)
        ra = save_payload(run, "sglang", schema.LAYER_INPUT, t)
        rb = save_payload(run, "megatron", schema.LAYER_INPUT, t)
        result = align_records([ra], [rb], run.run_dir, run.run_dir)
        assert len(result.matched) == 1
        assert result.matched[0].expert_id is None
        # the None group is not part of the per-expert bookkeeping
        assert result.one_to_one_per_expert == {}


class TestCosineFallback:
    def _near_but_not_equal(self, torch):
        g = torch.Generator().manual_seed(0)
        ta = torch.randn(64, generator=g)
        tb = ta + 1e-4 * torch.randn(64, generator=g)
        assert not torch.equal(ta, tb)
        cos = float(
            torch.dot(ta.double(), tb.double())
            / (ta.double().norm() * tb.double().norm())
        )
        assert cos >= COSINE_FALLBACK_THRESHOLD
        return ta, tb

    def test_fallback_disabled_leaves_records_unmatched(self, run):
        torch = require_torch()
        ta, tb = self._near_but_not_equal(torch)
        ra = save_payload(run, "sglang", schema.FC1_OUTPUT, ta, expert_id=0)
        rb = save_payload(run, "megatron", schema.FC1_OUTPUT, tb, expert_id=0)
        result = align_records([ra], [rb], run.run_dir, run.run_dir)
        assert result.matched == []
        assert result.unmatched_a == [ra]
        assert result.unmatched_b == [rb]
        assert result.min_fallback_cosine is None

    def test_fallback_enabled_matches_and_reports_cosine(self, run):
        torch = require_torch()
        ta, tb = self._near_but_not_equal(torch)
        ra = save_payload(run, "sglang", schema.FC1_OUTPUT, ta, expert_id=0)
        rb = save_payload(run, "megatron", schema.FC1_OUTPUT, tb, expert_id=0)
        result = align_records(
            [ra], [rb], run.run_dir, run.run_dir,
            allow_value_match_fallback=True,
        )
        assert result.method == METHOD_COSINE_FALLBACK
        assert len(result.matched) == 1
        assert result.matched[0].method == METHOD_COSINE_FALLBACK
        assert result.min_fallback_cosine is not None
        assert result.min_fallback_cosine >= COSINE_FALLBACK_THRESHOLD
        assert result.one_to_one_per_expert == {0: True}
