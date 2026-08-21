"""Explicit csrc source manifest for the fp4_gemm CUDA extension.

setup.py consumes this list instead of recursively globbing every .cpp/.cu.
New kernel files must be added here; a glob-vs-manifest check fails the build
if a translation unit is left out.
"""

from __future__ import annotations

import glob
import os
import pathlib

CSRC_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = CSRC_DIR.parent.parent  # repo root


def _rel(*parts: str) -> str:
    return os.path.relpath(CSRC_DIR.joinpath(*parts), ROOT_DIR)


# Runtime support compiled into the extension (not Python bindings).
COMMON_CPP = [
    _rel("util", "transformer_engine", "cuda_runtime.cpp"),
]

UTIL_SOURCES = [
    _rel("ops", "swizzle.cu"),
    _rel("util", "transformer_engine", "common.cu"),
]

FP4_QUANTIZE_SOURCES = [
    _rel("ops", "fp4", "quantization", "quantize_transpose_nvfp4.cu"),
    _rel("ops", "fp4", "quantization", "quantize_round_four_six.cu"),
    _rel("ops", "fp4", "quantization", "quantize_round_eden_fp4.cu"),
    _rel("ops", "fp4", "quantization", "quantize_pertoken_fp4.cu"),
    _rel("ops", "fp4", "quantization", "quantize_pertoken_fp4_fused.cu"),
    _rel("ops", "fp4", "quantization", "group_quantize_pertoken_fp4.cu"),
    _rel("ops", "fp4", "quantization", "group_quantize_pertoken_fp4_split_amax.cu"),
    _rel("ops", "fp4", "quantization", "group_transform_and_eden.cu"),
    _rel("ops", "fp4", "quantization", "quantize_rht128_eden.cu"),
    _rel("ops", "fp4", "quantization", "quantize_rht128_requant.cu"),
    _rel("ops", "fp4", "quantization", "dq_tp_q.cu"),
]

FP4_GEMM_SOURCES = [
    _rel("ops", "fp4", "gemm", "gemm_cublas.cu"),
    _rel("ops", "fp4", "gemm", "gemm_cutlass.cu"),
    _rel("ops", "fp4", "gemm", "gemm_grouped_cutlass.cu"),
    _rel("ops", "fp4", "gemm", "gemm_grouped_cutlass_v2.cu"),
    _rel("ops", "fp4", "gemm", "gemm_grouped_cutlass_v2_evt.cu"),
]

FP8_QUANTIZE_SOURCES = [
    _rel("ops", "fp8", "quantization", "group_quantize_mxfp8.cu"),
    _rel("ops", "fp8", "quantization", "group_quantize_mxfp8_chunked.cu"),
    _rel("ops", "fp8", "quantization", "group_transpose_quantize_mxfp8.cu"),
]

FP8_GEMM_SOURCES = [
    _rel("ops", "fp8", "gemm", "gemm_mxfp8_grouped_cutlass.cu"),
]

BINDING_SOURCES = [
    _rel("torch_bindings.cpp"),
    _rel("bindings", "register_utils.cpp"),
    _rel("bindings", "register_fp4_quantize.cpp"),
    _rel("bindings", "register_fp4_gemm.cpp"),
    _rel("bindings", "register_fp8_quantize.cpp"),
    _rel("bindings", "register_fp8_gemm.cpp"),
    _rel("bindings", "register_fp4_quarter.cpp"),
]


def source_groups() -> dict[str, list[str]]:
    return {
        "common": COMMON_CPP,
        "util": UTIL_SOURCES,
        "fp4_quant": FP4_QUANTIZE_SOURCES,
        "fp4_gemm": FP4_GEMM_SOURCES,
        "fp8_quant": FP8_QUANTIZE_SOURCES,
        "fp8_gemm": FP8_GEMM_SOURCES,
        "bindings": BINDING_SOURCES,
    }


def all_csrc_sources() -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for group in source_groups().values():
        for path in group:
            if path in seen:
                raise RuntimeError(f"duplicate csrc source in manifest: {path}")
            seen.add(path)
            sources.append(path)
    return sources


def discover_glob_sources() -> list[str]:
    """Filesystem scan used only to detect files missing from the manifest."""
    patterns = [
        str(CSRC_DIR / "**" / "*.cpp"),
        str(CSRC_DIR / "**" / "*.cu"),
    ]
    found = []
    for pattern in patterns:
        found.extend(os.path.relpath(p, ROOT_DIR) for p in glob.glob(pattern, recursive=True))
    return sorted(found)


def assert_manifest_covers_glob() -> None:
    """Fail the build if a .cpp/.cu exists under csrc but is not listed."""
    manifest = set(all_csrc_sources())
    discovered = set(discover_glob_sources())
    missing = sorted(discovered - manifest)
    extra = sorted(manifest - discovered)
    if missing:
        raise RuntimeError(
            "csrc source manifest is missing files found on disk:\n  "
            + "\n  ".join(missing)
            + "\nAdd them to miniTransformer/csrc/source_manifest.py"
        )
    if extra:
        raise RuntimeError(
            "csrc source manifest lists files that do not exist:\n  "
            + "\n  ".join(extra)
        )


def format_source_stats() -> str:
    groups = source_groups()
    parts = [f"{name}={len(files)}" for name, files in groups.items()]
    total = sum(len(files) for files in groups.values())
    return " ".join(parts) + f" total={total}"


if __name__ == "__main__":
    assert_manifest_covers_glob()
    print("[mini-TE] csrc sources:", format_source_stats())
    for path in all_csrc_sources():
        print(path)
