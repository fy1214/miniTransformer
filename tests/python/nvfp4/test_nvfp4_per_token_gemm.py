import pytest
import torch

pytest.importorskip("cutlass")
pytest.importorskip("fp4_gemm")

import miniTransformer.ops.nvfp4_qgemm as nvfp4_qgemm


def _make_inputs(m=8, n=16, k=64):
    return (
        torch.empty((m, k // 2), dtype=torch.uint8),
        torch.empty((n, k // 2), dtype=torch.uint8),
        torch.empty((m, k // 16), dtype=torch.float8_e4m3fn),
        torch.empty((n, k // 16), dtype=torch.float8_e4m3fn),
    )


def test_dense_nvfp4_per_token_gemm_scalar_alpha(monkeypatch):
    called = {}

    def _fake_grouped(**kwargs):
        called["alpha_shape"] = tuple(kwargs["alpha_list"][0].shape)
        called["out_dtype"] = kwargs["out_dtype"]
        return ["ok_scalar"]

    monkeypatch.setattr(nvfp4_qgemm, "grouped_nvfp4_per_token_gemm", _fake_grouped)
    a, b, a_sf, b_sf = _make_inputs()
    ret = nvfp4_qgemm.dense_nvfp4_per_token_gemm(
        a, b, a_sf, b_sf, alpha=torch.tensor([1.0], dtype=torch.float32)
    )
    assert ret == "ok_scalar"
    assert called["alpha_shape"] == (1,)
    assert called["out_dtype"] == torch.bfloat16


def test_dense_nvfp4_per_token_gemm_per_row_alpha(monkeypatch):
    called = {}

    def _fake_grouped(**kwargs):
        called["alpha_shape"] = tuple(kwargs["alpha_list"][0].shape)
        return ["ok_row"]

    monkeypatch.setattr(nvfp4_qgemm, "grouped_nvfp4_per_token_gemm", _fake_grouped)
    a, b, a_sf, b_sf = _make_inputs(m=10)
    alpha = torch.ones((10,), dtype=torch.float32)
    ret = nvfp4_qgemm.dense_nvfp4_per_token_gemm(a, b, a_sf, b_sf, alpha=alpha)
    assert ret == "ok_row"
    assert called["alpha_shape"] == (10,)


def test_dense_nvfp4_per_token_gemm_invalid_alpha_shape():
    a, b, a_sf, b_sf = _make_inputs(m=10)
    with pytest.raises(ValueError, match="alpha must be shape"):
        nvfp4_qgemm.dense_nvfp4_per_token_gemm(
            a,
            b,
            a_sf,
            b_sf,
            alpha=torch.ones((3,), dtype=torch.float32),
        )


def test_grouped_nvfp4_per_token_gemm_dispatch(monkeypatch):
    called = {}

    def _fake_run(**kwargs):
        called["out_dtype"] = kwargs["out_dtype"]
        called["num_groups"] = len(kwargs["abc_tensors"]["torch"])
        return ["ok_grouped"]

    class _FakeKernel:
        def run_from_grouped_inputs(self, **kwargs):
            return _fake_run(**kwargs)

    monkeypatch.setattr(
        nvfp4_qgemm,
        "Sm100GroupedBlockScaledGemmKernel",
        lambda *args, **kwargs: _FakeKernel(),
    )
    a, b, a_sf, b_sf = _make_inputs()
    ret = nvfp4_qgemm.grouped_nvfp4_per_token_gemm(
        a_list=[a],
        b_list=[b],
        a_descale_list=[a_sf],
        b_descale_list=[b_sf],
        alpha_list=[torch.tensor([1.0], dtype=torch.float32)],
    )
    assert ret == ["ok_grouped"]
    assert called["out_dtype"] == torch.bfloat16
    assert called["num_groups"] == 1
