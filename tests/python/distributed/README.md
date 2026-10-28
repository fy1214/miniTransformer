# Distributed Parallel Strategy Tests

Correctness scripts for miniTransformer TP/SP (Tensor Parallel + Sequence Parallel)
and CuTeDSL GEMM-AllReduce. All multi-rank cases launch with `torchrun`.

## Layout

| Path | Entry | What it checks |
|---|---|---|
| `test_allreduce.py` | `__main__` | 2-rank AllReduce: complementary shards sum to full tensor |
| `test_row_parallel_gemm.py` | `__main__` | Row-parallel GEMM + AllReduce (integer bit-exact + precision variants) |
| `test_linear_tp.py` | `run_tests()` | miniTransformer Linear column/row TP ± SP (default BF16_REF) |
| `test_linear_tp_quantized.py` | `run_tests()` | Quantized Linear TP with shared weight-cache inject (bit-exact) |
| `test_te_linear_nvfp4_tp.py` | `run_tests()` | TE Linear TP communication in bf16 (filename historical; no NVFP4 recipe) |
| `cutedsl/test_gemm_allreduce.py` | `run_tests()` | `GemmAllReduceCtx` vs torch GEMM+AllReduce |
| `cutedsl/benchmark/*` | `__main__` | Manual bench / precision / nsys·ncu for GEMM+AllReduce |
| `helpers.py` | — | Shared torchrun init / TP group / broadcast |
| `cutedsl/helpers.py` | — | Shared NVSHMEM init + schemes A/B/C |

## Running

```bash
# AllReduce primitive
torchrun --nproc_per_node=2 test_allreduce.py

# Row parallel GEMM
torchrun --nproc_per_node=2 test_row_parallel_gemm.py

# miniTransformer Linear TP-SP (BF16 reference by default)
torchrun --nproc_per_node=2 test_linear_tp.py --qparam 6 --tp-size 2
torchrun --nproc_per_node=2 test_linear_tp.py --qparam 6 --tp-size 2 --no-sp

# Quantized mini Linear TP
torchrun --nproc_per_node=2 test_linear_tp_quantized.py
torchrun --nproc_per_node=2 test_linear_tp_quantized.py --no-sp --qparam fp4

# TE Linear TP (bf16)
torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py
torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py --no-sp

# CuTeDSL GemmAllReduce
torchrun --nproc_per_node=2 cutedsl/test_gemm_allreduce.py
```

`--qparam` for `test_linear_tp.py`:

| Value | Recipe |
|---|---|
| 1 | NON_QUANTIZE |
| 2 | INT4_SUB_CHANNEL |
| 3 | FP4_SUB_CHANNEL_REF |
| 4 | FP4_SUB_CHANNEL |
| 5 | FP4_SUB_CHANNEL_QUARTET |
| 6 | BF16_REF (default) |

## Verification notes

Integer inputs `x, w, dy ∈ {-1, 0, 1}` keep products/sums exact in bf16 for modest `K`,
so several TP checks use `rtol=0, atol=0`.

Column-parallel forward with random bf16 inputs typically uses `rtol=atol=1e-2`.

## TP-SP math (dense Linear)

### Column parallel

`W` is split on `out_features`. With SP, each rank AllGathers its sequence shard before GEMM.
Backward: dgrad AllReduce (then ReduceScatter if SP); wgrad stays local per out-shard.

### Row parallel

`W` and `x` are split on `in_features`. Forward AllReduce (or ReduceScatter if SP).
Backward: dgrad is local per in-shard; wgrad may AllGather `dy` when SP is on.
