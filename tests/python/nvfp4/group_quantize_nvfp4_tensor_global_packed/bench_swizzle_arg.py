#!/usr/bin/env python3
"""
Benchmark slime weight-update quantize path: baseline (swizzle=True + Python
_unswizzle_scale_to_rowmajor) vs new (swizzle=False).

Shapes mirror typical MoE expert weights for fine-grained MoE:
  - Qwen3-30B-A3B   gate/up: [768, 2048],    down: [2048, 768]
  - DeepSeek-V3     gate/up: [2048, 7168],   down: [7168, 2048]

Run (B300 / GPU box):
  cd tests/python/nvfp4/group_quantize_nvfp4_tensor_global_packed
  python3 bench_swizzle_arg.py
"""
import torch

import fp4_gemm
from miniTransformer.ops.nvfp4_quantize import _unswizzle_scale


SHAPES = [
    ("Qwen3-30B-A3B gate/up", 768, 2048),
    ("Qwen3-30B-A3B down",   2048, 768),
    ("DeepSeek-V3 gate/up",  2048, 7168),
    ("DeepSeek-V3 down",     7168, 2048),
]
WARMUP = 20
ITERS  = 200
T      = 128


def _run(rows, cols, swizzle, do_unswizzle):
    device = "cuda"
    n_padded = ((rows + T - 1) // T) * T
    sf_stride = n_padded * (cols // 16)

    x = torch.randn(rows, cols, dtype=torch.bfloat16, device=device) * 4.0
    row_offsets = torch.tensor([0, rows], dtype=torch.int32, device=device)
    scale_offsets = torch.tensor([0, sf_stride], dtype=torch.int64, device=device)
    amax = x.abs().max().to(torch.float32).reshape(1).contiguous()

    data = torch.empty((rows, cols // 2), dtype=torch.uint8, device=device)
    gs = torch.empty(1, dtype=torch.float32, device=device)
    sf = torch.zeros(sf_stride, dtype=torch.float8_e4m3fn, device=device)

    def _one_call():
        fp4_gemm.rtn_fp4_group_tensor_global(
            data, sf, gs, x, row_offsets, scale_offsets, amax, 1.0, swizzle=swizzle,
        )
        if do_unswizzle:
            _ = _unswizzle_scale(sf.to(torch.float32), n_padded, cols)[:rows]
        else:
            _ = sf.view(n_padded, cols // 16)[:rows]

    for _ in range(WARMUP):
        _one_call()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        _one_call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS  # ms / call


def main():
    print(f"{'shape':<27} {'rows':>6} {'cols':>6}  {'baseline (ms)':>14}  {'new (ms)':>10}  {'speedup':>8}")
    print("-" * 78)
    for label, rows, cols in SHAPES:
        baseline = _run(rows, cols, swizzle=True,  do_unswizzle=True)
        new      = _run(rows, cols, swizzle=False, do_unswizzle=False)
        speedup  = baseline / new if new > 0 else float("inf")
        print(f"{label:<27} {rows:>6} {cols:>6}  {baseline:>14.4f}  {new:>10.4f}  {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
