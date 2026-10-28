# NVFP4 tests (`tests/python/nvfp4`)

Tests for NVFP4 quantize / dequant / grouped GEMM on Blackwell (sm_100+).

## Layout

```
tests/python/nvfp4/
├── conftest.py              # sm_100 guards, fp4_gemm fixture, pytest markers
├── helpers/
│   ├── golden.py            # unpack / unswizzle / dequant / eff-bitwidth
│   ├── metrics.py           # real-input eval metrics (load, compute, print)
│   └── shapes.py            # shared MoE / GEMM shape constants
├── quantize_nvfp4_pertoken/           # single-matrix pertoken quant
├── group_quantize_nvfp4_pertoken_swizzled/  # MoE activation quant
├── group_quantize_nvfp4_tensor_global_packed/  # MoE weight quant
├── grouped_cutlass_gemm/              # CUTLASS v2 grouped GEMM
└── test_*.py                          # top-level pytest (pertoken GEMM, compile cache, …)
```

**Shared helpers:** import from the nvfp4 test root (pytest adds it via `conftest.py`):

```python
from helpers.golden import unpack_fp4, unswizzle_scale, effective_bitwidth
from helpers.metrics import load_real_input, run_quant_eval
from helpers.shapes import FC1_DUMP_PATH, GROUPED_GEMM_V2_CASES
```

Legacy `*/real_input_eval_utils.py` files re-export `helpers.metrics` for backward compatibility.

## Pytest markers

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.production` | MoE GroupedLinear forward path (QAT 6–15) |
| `@pytest.mark.legacy` | Dense GEMM, TE, quartet, exploratory backends |
| `@pytest.mark.eval` | Needs real fc1 dump (`/tmp/dump/fc1_step0000_inp.pt`) |
| `@pytest.mark.bench` | Performance only, not correctness |
| `@pytest.mark.slow` | Long-running (also registered in parent conftest) |

Markers are registered in `conftest.py` and applied on remaining tests (P2):
- `production` — MoE quant/GEMM mainline + pad-fuse API
- `legacy` — TE tiled quant / binding smoke / CuTeDSL blockscaled kernel
- `bench`+`slow` — micro-benches inside pytest files

Example: `pytest tests/python/nvfp4 -m "production and not bench" -v`

## Fixtures (`conftest.py`)

| Fixture | Skips when |
|---------|------------|
| `require_cuda` | No CUDA |
| `require_sm100` | GPU major < 10 |
| `require_fp4_gemm` | Extension missing or sm_100 unavailable |
| `fp4_gemm_module` | Same as `require_fp4_gemm`, returns `fp4_gemm` module |

Run from repo root:

```bash
pytest tests/python/nvfp4/quantize_nvfp4_pertoken/test_nvfp4_pertoken.py -v
pytest tests/python/nvfp4/grouped_cutlass_gemm/ -v
pytest tests/python/nvfp4 --nvfp4-cuda-device cuda:0 -k "pertoken" -v
```

## QAT_PARAMS → production API → tests

Default training recipe: `QAT_PARAMS=6` (`FP4_SUB_CHANNEL_CURSOR2_REF`).

| QAT | Recipe | Quant (X / W) | GEMM | Primary tests |
|-----|--------|---------------|------|----------------|
| 3 | `FP4_SUB_CHANNEL_REF` | TE tiled ref | TE / ref | `test_nvfp4_quantize_exact.py` |
| 4–5 | ref / quartet | legacy | — | *(removed — no dedicated tests)* |
| **6** | **`FP4_SUB_CHANNEL_CURSOR2_REF`** | **`quantize_nvfp4_pertoken*`**, **`group_quantize_nvfp4_*`** | CuTeDSL ref | **`quantize_nvfp4_pertoken/`**, **`group_quantize_nvfp4_*/`** |
| 7 | `FP4_SUB_CHANNEL_CURSOR2` | same quant ops | CuTeDSL | `test_grouped_gemm_pertoken.py` |
| **8** | **`FP4_SUB_CHANNEL_CURSOR2_CUTLASS`** | group quant | **`grouped_cutlass_gemm_v2`** | **`grouped_cutlass_gemm/test_grouped_cutlass_gemm_v2_vs_golden.py`** |
| 12 | `CURSOR2_CUTLASS_BWD` | group quant | CUTLASS v2 | `grouped_cutlass_gemm/` |
| **14** | **`CURSOR2_CUTLASS_V2`** | group quant | **`grouped_cutlass_gemm_v2_evt`** | **`grouped_cutlass_gemm/test_grouped_cutlass_gemm_v2_evt_vs_golden.py`** |
| **15** | **`FP4_SUB_CHANNEL_PERTOKEN`** | **`group_quantize_nvfp4_pertoken_split_amax`** | CUTLASS v2 | **missing — add in P5** |

## File index

### Production quant (MoE)

| File | API under test | Compare against |
|------|----------------|-----------------|
| `quantize_nvfp4_pertoken/test_nvfp4_pertoken.py` | `quantize_nvfp4_pertoken`, `_swizzled` | Python golden, fused vs two-pass |
| `group_quantize_nvfp4_pertoken_swizzled/test_nvfp4_group_pertoken_quantize.py` | `group_quantize_nvfp4_pertoken_swizzled`, `group_quantize_nvfp4_tensor_global_packed`, `quantize_weight_nvfp4_swizzled` | per-group ref |
| `group_quantize_nvfp4_pertoken_swizzled/test_nvfp4_group_pertoken_pad_fuse.py` | pad-fuse CUDA kernels | legacy grouped pertoken |
| `group_quantize_nvfp4_pertoken_swizzled/test_nvfp4_group_pertoken_pad_fuse_wrapper.py` | `fuse_pad=True` wrapper | direct kernel |
| `group_quantize_nvfp4_tensor_global_packed/test_swizzle_arg_parity.py` | `group_quantize_nvfp4_tensor_global_packed(swizzle=…)` | swizzle on/off parity |

### Production GEMM (MoE)

| File | API under test | Compare against |
|------|----------------|-----------------|
| `grouped_cutlass_gemm/test_grouped_cutlass_gemm_v2_vs_golden.py` | `grouped_cutlass_gemm_v2` | dequant-then-GEMM Python golden |
| `grouped_cutlass_gemm/test_grouped_cutlass_gemm_v2_evt_vs_golden.py` | v2 + EVT epilogue | v2 baseline |
| `test_grouped_gemm_pertoken.py` | CuTeDSL grouped pertoken GEMM | Python reference matmul |

### Other pytest

| File | Notes |
|------|-------|
| `test_nvfp4_quantize_exact.py` | legacy: TE tiled + CUDA transpose vs TE + binding smoke / `quantize_nvfp4_cuda` |
| `test_grouped_gemm_compile_cache.py` | compile cache helpers (no CUDA) |
| `test_group_gemm_blockscaled_nvfp4_mxfp_cute.py` | CuTeDSL blockscaled kernel dev |

### Real-input eval (manual, needs dump)

| File | Operator |
|------|----------|
| `quantize_nvfp4_pertoken/eval_nvfp4_pertoken_real_input.py` | single-matrix pertoken + weight swizzled |
| `group_quantize_nvfp4_pertoken_swizzled/eval_group_quantize_pertoken_swizzled_real_input.py` | grouped pertoken swizzled |
| `group_quantize_nvfp4_pertoken_swizzled/eval_pad_fuse_real_input.py` | pad fuse path |
| `group_quantize_nvfp4_tensor_global_packed/eval_group_quantize_tensor_global_packed_real_input.py` | grouped weight packed |
| `grouped_cutlass_gemm/eval_grouped_cutlass_gemm_v2_real_input.py` | end-to-end quant + GEMM v2 |

Collect dump:

```bash
export LINEAR_STATS_STEPS=1
export LINEAR_STATS_CLASS=grouped
export LINEAR_STATS_DUMP_DIR=/tmp/dump
export LINEAR_STATS_BWD=0
# run training / forward once → fc1_step0000_inp.pt
```

### Bench scripts (manual, not CI)

| File | Purpose |
|------|---------|
| `group_quantize_nvfp4_tensor_global_packed/bench_swizzle_arg.py` | swizzle arg micro-bench |
| `group_quantize_nvfp4_pertoken_swizzled/bench_nvfp4_group_pertoken_chunked_vs_legacy.py` | chunked vs legacy scheduler |

## Removed (2026-09 batch 1)

Deleted as unused or superseded by production v2 / grouped quant tests:

- Dense GEMM: `test_nvfp4_gemm_{te,cutlass,cute,triton}.py`
- Legacy paths: `test_nvfp4_quartet_*.py`, `test_grouped_gemm_nvfp4_vs_bf16_diff.py`
- Non-NVFP4: `test_group_gemm_fp16_bf16_cute.py`, `test_group_gemm_blockwise_fp8_cute.py`
- No-value: `test_nvfp4_per_token_gemm.py`, `test_quantize_pertoken_triton.py`
- CUTLASS v1 (dead API): `grouped_cutlass_gemm/test_grouped_cutlass_gemm_per_token_scale.py`
- Profile/bench tools: `profile_*.py`, `bench_te_per_token_quant_paths.py`

## Subdirectory READMEs

- `quantize_nvfp4_pertoken/README.md` — QAT 6 single-matrix quant
- `group_quantize_nvfp4_pertoken_swizzled/README.md` — MoE activation quant
- `group_quantize_nvfp4_tensor_global_packed/README.md` — MoE weight quant
- `grouped_cutlass_gemm/README.md` — CUTLASS v2 golden methodology

## Next steps

| Phase | Work |
|-------|------|
| ~~P2~~ | ~~Add docstrings + `@pytest.mark.*`~~ (done) |
| ~~P3~~ | ~~Merge binding into `test_nvfp4_quantize_exact.py`~~ (done) |
| P4 | Move bench scripts to `tools/`, trim `test_bench_*` inside pertoken tests |
| P5 | Add `group_quantize_nvfp4_pertoken_split_amax` tests (QAT 15) |
