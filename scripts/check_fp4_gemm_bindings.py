#!/usr/bin/env python3
"""Check fp4_gemm public binding names and the csrc source manifest.

The manifest check always runs. The live-module check runs only when
``fp4_gemm`` can be imported (requires a built CUDA extension).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

EXPECTED_BINDINGS = {
    "quantize_transpose_nvfp4",
    "swizzle_scaling_factors",
    "cublas_gemm",
    "block_scaled_matmul_nvfp4",
    "cutlass_gemm",
    "grouped_cutlass_gemm_with_per_token_scale",
    "grouped_cutlass_gemm_v2_buffer_sizes",
    "grouped_cutlass_gemm_v2",
    "grouped_cutlass_gemm_v2_evt_buffer_sizes",
    "grouped_cutlass_gemm_v2_evt",
    "mxfp8_grouped_gemm_buffer_sizes",
    "mxfp8_grouped_gemm",
    "mxfp8_grouped_quantize",
    "mxfp8_grouped_quantize_hw",
    "mxfp8_grouped_transpose_quantize",
    "mxfp8_grouped_quantize_chunked",
    "mxfp8_grouped_quantize_chunked_hw",
    "four_six_fp4",
    "rtn_fp4",
    "four_six_fp4_pertoken",
    "rtn_fp4_pertoken",
    "four_six_fp4_pertoken_linear",
    "rtn_fp4_pertoken_linear",
    "four_six_fp4_pertoken_fused",
    "rtn_fp4_pertoken_fused",
    "four_six_fp4_pertoken_linear_fused",
    "rtn_fp4_pertoken_linear_fused",
    "rtn_fp4_group_pertoken",
    "four_six_fp4_group_pertoken",
    "rtn_fp4_group_pertoken_legacy_per_row",
    "four_six_fp4_group_pertoken_legacy_per_row",
    "rtn_fp4_group_pertoken_pad_fuse",
    "four_six_fp4_group_pertoken_pad_fuse",
    "rtn_fp4_group_tensor_global",
    "four_six_fp4_group_tensor_global",
    "eden_fp4",
    "group_transform_128_eden",
    "dequant_tp_had_quant",
    "rht_128_eden",
    "rht128_requant",
    "rtn_fp4_group_pertoken_split_amax",
    "four_six_fp4_group_pertoken_split_amax",
    "fp4_perrow_amax_te",
}


def _load_sources():
    path = ROOT / "miniTransformer" / "csrc" / "source_manifest.py"
    spec = importlib.util.spec_from_file_location("mini_te_csrc_sources", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_manifest() -> None:
    sources = _load_sources()
    sources.assert_manifest_covers_glob()
    print("[ok] csrc source manifest matches glob:", sources.format_source_stats())


def check_live_module() -> None:
    try:
        import fp4_gemm
    except ImportError as exc:
        print(f"[skip] fp4_gemm not importable ({exc})")
        return

    exported = {name for name in dir(fp4_gemm) if not name.startswith("_")}
    missing = sorted(EXPECTED_BINDINGS - exported)
    extra = sorted(exported - EXPECTED_BINDINGS)
    if missing:
        raise SystemExit("fp4_gemm is missing bindings:\n  " + "\n  ".join(missing))
    if extra:
        print("[warn] fp4_gemm has extra public names:\n  " + "\n  ".join(extra))
    print(f"[ok] fp4_gemm exports {len(EXPECTED_BINDINGS)} expected bindings")


def main() -> int:
    check_manifest()
    check_live_module()
    return 0


if __name__ == "__main__":
    sys.exit(main())
