# Distributed Parallel Strategy Tests

Correctness tests for miniTransformer's TP-SP (Tensor Parallel + Sequence Parallel) strategy.
All tests use `torchrun` for multi-process launch and verify mathematical equivalence against a
single-rank baseline.

## Files

| File | Entry point | What it tests |
|---|---|---|
| `test_allreduce.py` | `run_allreduce_test()` | AllReduce primitive: t1 + t2 == t0 after reduction |
| `test_row_parallel_gemm.py` | `run_row_parallel_gemm_test()` | Row parallel GEMM + AllReduce, bit-exact (integer inputs) |
| `test_fp32_gemm_allreduce.py` | `run_tests()` | fp32 GEMM + fp32 AllReduce fusion vs bf16 variants |
| `test_linear_tp.py` | `run_tests()` | miniTransformer Linear column/row parallel forward + backward |
| `test_te_linear_nvfp4_tp.py` | `run_tests()` | TransformerEngine Linear TP communication (column/row parallel) |

## Running

```bash
# AllReduce primitive
torchrun --nproc_per_node=2 test_allreduce.py

# Row parallel GEMM
torchrun --nproc_per_node=2 test_row_parallel_gemm.py

# fp32 GEMM + AllReduce precision comparison
torchrun --nproc_per_node=2 test_fp32_gemm_allreduce.py

# miniTransformer Linear TP-SP (BF16 reference)
torchrun --nproc_per_node=2 test_linear_tp.py --qparam 6 --tp-size 2
torchrun --nproc_per_node=2 test_linear_tp.py --qparam 6 --tp-size 2 --no-sp

# TE Linear TP-SP
torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py
torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py --no-sp
```

`--qparam` values for `test_linear_tp.py`:

| Value | Recipe |
|---|---|
| 1 | NON_QUANTIZE |
| 2 | INT4_SUB_CHANNEL |
| 3 | FP4_SUB_CHANNEL_REF |
| 4 | FP4_SUB_CHANNEL |
| 5 | FP4_SUB_CHANNEL_QUARTET |
| 6 | BF16_REF (default) |

## Verification strategy

Integer-valued inputs `x, w ∈ {-1, 0, 1}`, `dy ∈ {-1, 0, 1}` are used for backward tests.
Products and accumulated sums are integers; bf16 represents integers exactly in `[-2048, 2048]`,
so `K=4096` accumulations stay in range and allow bit-exact verification (`rtol=0, atol=0`).

Forward tests (column parallel) use `rtol=atol=1e-2` because random bf16 inputs are used.

## TP-SP math

### Column parallel linear

`W` is split column-wise: each rank `r` holds `W_r = W[r*out//tp : (r+1)*out//tp, :]`

**SP=False** — every rank receives the full input `x [M, K]`:
```
out_r [M, out//tp] = x @ W_r.T
out   [M, out]     = concat(out_0, ..., out_{tp-1})   # no communication needed
```

**SP=True** — each rank receives a sequence shard `x_r [M//tp, K]`;
the linear layer AllGathers to restore `x [M, K]` before the GEMM:
```
out_r [M, out//tp] = AllGather(x_r) @ W_r.T           # same result as SP=False
```

**Backward:**
- `dx_r = dy_r @ W_r.T` then AllReduce → full `dx [M, K]`
  - SP=True: additional ReduceScatter → each rank holds `dx_r [M//tp, K]`
- `dW_r = dy_r.T @ x` — independent per rank, no communication

### Row parallel linear

`W` is split row-wise (along `in_features`): rank `r` holds `W_r = W[:, r*in//tp : (r+1)*in//tp]`.
Input is also split accordingly: `x_r = x[:, r*in//tp : (r+1)*in//tp]`.

**SP=False** — each rank computes a partial output, then AllReduce:
```
out_r [M, N] = x_r @ W_r.T
out   [M, N] = AllReduce(out_0 + ... + out_{tp-1})
```

**SP=True** — AllReduce is replaced by ReduceScatter, distributing output over sequence dim:
```
out_r [M//tp, N] = ReduceScatter(out_0 + ... + out_{tp-1})
```

**Backward:**
- `dx_r = dy @ W_r.T` — each rank computes its own `in_features` shard directly
  - SP=True: AllGather `dy_r` → full `dy [M, N]` before computing `dx_r`
- `dW_r = dy.T @ x_r` — independent per rank
