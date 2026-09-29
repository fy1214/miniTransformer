"""Tests for logpdiff.schema: TensorRecord round-trip and validation."""

from __future__ import annotations

import json

import pytest

from logpdiff import schema
from logpdiff.schema import TensorRecord, check_single_device_parallel

from conftest import make_record


def _valid_record(**kwargs):
    fields = dict(
        run_id="run-1",
        framework="sglang",
        sample_id="sample-0",
        layer_id=0,
        tensor_point=schema.FC1_OUTPUT,
        shape=(2, 8),
        dtype="torch.float32",
        provenance=schema.ACTUAL_KERNEL,
        payload_path="sglang/tensors/x.pt",
        global_rank=0,
        expert_id=3,
        token_position=7,
        route_slot=1,
    )
    fields.update(kwargs)
    return TensorRecord(**fields)


class TestRoundTrip:
    def test_full_record_round_trip(self):
        rec = _valid_record()
        restored = TensorRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
        assert restored == rec
        assert restored.shape == (2, 8)  # list -> tuple normalization

    def test_optional_fields_round_trip(self):
        rec = _valid_record(
            global_rank=None, expert_id=None, token_position=None,
            route_slot=None, shape=(),
        )
        restored = TensorRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
        assert restored == rec
        assert restored.expert_id is None

    def test_int_and_str_sample_id(self):
        for sample_id in (0, 17, "sample-0"):
            rec = _valid_record(sample_id=sample_id)
            rec.validate()
            assert TensorRecord.from_dict(rec.to_dict()).sample_id == sample_id

    def test_from_dict_rejects_unknown_fields(self):
        d = _valid_record().to_dict()
        d["surprise"] = 1
        with pytest.raises(ValueError, match="unknown TensorRecord fields"):
            TensorRecord.from_dict(d)


class TestValidate:
    def test_valid_record_passes(self):
        _valid_record().validate()

    @pytest.mark.parametrize("provenance", sorted(schema.PROVENANCE_ALL))
    def test_all_known_provenances_accepted(self, provenance):
        _valid_record(provenance=provenance).validate()
        assert schema.validate_provenance(provenance) == provenance

    def test_bad_provenance_rejected(self):
        with pytest.raises(ValueError, match="unknown provenance"):
            _valid_record(provenance="kernel_guess").validate()
        with pytest.raises(ValueError):
            schema.validate_provenance("actual")  # alias is not a provenance

    def test_bad_tensor_point_rejected(self):
        with pytest.raises(ValueError, match="unknown tensor_point"):
            _valid_record(tensor_point="fc1_gemm").validate()
        with pytest.raises(ValueError):
            schema.validate_tensor_point("not_a_point")

    @pytest.mark.parametrize("shape", [(-1, 4), (1.5, 4), (True, 4), ("8",)])
    def test_bad_shape_rejected(self, shape):
        with pytest.raises(ValueError):
            _valid_record(shape=shape).validate()

    def test_shape_list_is_normalized_to_tuple(self):
        rec = _valid_record(shape=[2, 8])
        assert rec.shape == (2, 8)  # __post_init__ normalizes
        rec.validate()

    @pytest.mark.parametrize(
        "field,bad",
        [
            ("run_id", ""),
            ("framework", ""),
            ("dtype", ""),
            ("payload_path", ""),
            ("layer_id", -1),
            ("layer_id", True),
            ("global_rank", -1),
            ("expert_id", -2),
            ("token_position", -3),
            ("route_slot", -4),
        ],
    )
    def test_field_validation(self, field, bad):
        with pytest.raises(ValueError):
            _valid_record(**{field: bad}).validate()

    @pytest.mark.parametrize("sample_id", [None, ""])
    def test_sample_id_must_be_set(self, sample_id):
        with pytest.raises(ValueError):
            _valid_record(sample_id=sample_id).validate()

    def test_make_record_helper_is_valid(self):
        make_record("r", "megatron", schema.FC1_QDATA).validate()


class TestParallelGuard:
    def test_single_device_ok(self):
        check_single_device_parallel(1, 1, 1)
        check_single_device_parallel()  # defaults

    @pytest.mark.parametrize(
        "tp,ep,pp", [(2, 1, 1), (1, 2, 1), (1, 1, 2), (8, 8, 4), (0, 1, 1)]
    )
    def test_multi_device_rejected(self, tp, ep, pp):
        with pytest.raises(ValueError, match="TP=EP=PP=1"):
            check_single_device_parallel(tp, ep, pp)


class TestPipelineStages:
    def test_stages_cover_expected_order(self):
        stages = schema.PIPELINE_STAGES
        assert stages[0] == "layer_input"
        assert stages[-1] == "logprob"
        assert stages.index("routing") < stages.index("dispatch")
        assert stages.index("fc1_quant") < stages.index("fc1_gemm")
        assert stages.index("fc2_quant") < stages.index("fc2_gemm")
        assert stages.index("fc1_gemm") < stages.index("swiglu")
        assert stages.index("swiglu") < stages.index("fc2_quant")
