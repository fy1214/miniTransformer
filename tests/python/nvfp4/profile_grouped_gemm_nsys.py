"""
nsys profiling script for Sm100GroupedBlockScaledGemmKernel.

Run with:
    nsys profile --trace=cuda,nvtx,osrt -o grouped_gemm_nsys --force-overwrite true \
        python tests/python/nvfp4/profile_grouped_gemm_nsys.py

Then inspect:
    nsys stats grouped_gemm_nsys.nsys-rep
    nsys-ui grouped_gemm_nsys.nsys-rep   # GUI

Key things to look for in the nsys timeline:
  - GPU timeline: how long does the actual CUDA kernel run?
  - CPU→GPU gap: how long is the GPU idle between kernel launches?
  - CUDA memcpy: are there H2D copies on every call?
  - NVTX ranges: which Python phase is the bottleneck?
"""
import time

import torch
import torch.cuda.nvtx as nvtx

from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute_alpha_epilogue import (
    GroupedBlockScaledProfile,
    Sm100GroupedBlockScaledGemmKernel,
    generate_grouped_blockscaled_inputs,
)

PROFILE = GroupedBlockScaledProfile.NVFP4

# Typical MoE shape: 8 experts
M, N, K, G = 128, 4096, 7168, 8
m_list = [M] * G
n_list = [N] * G
k_list = [K] * G

# ── Generate inputs once (not part of kernel timing) ─────────────────────────
nvtx.range_push("generate_inputs")
abc, sfasfb, problem_sizes = generate_grouped_blockscaled_inputs(
    m_list=m_list, n_list=n_list, k_list=k_list,
    profile=PROFILE, device="cuda", out_dtype=torch.bfloat16,
)
nvtx.range_pop()

device = abc["torch"][0][0].device
kernel = Sm100GroupedBlockScaledGemmKernel(profile=PROFILE)


def _make_alphas_packed():
    """Simulate per-token alpha from grouped_gemm_cutedsl_pertoken."""
    alpha_b_per_group = [0.01] * len(problem_sizes)
    global_scales = torch.cat([
        torch.rand(m, dtype=torch.float32, device=device)
        for m, *_ in problem_sizes
    ])
    alpha_b_vec = torch.tensor(alpha_b_per_group, dtype=torch.float32, device=device)
    repeats = torch.tensor([ps[0] for ps in problem_sizes], dtype=torch.int64, device=device)
    return global_scales * alpha_b_vec.repeat_interleave(repeats)


# ── Helper ───────────────────────────────────────────────────────────────────
def call_kernel(label, alphas=None, alphas_packed=None):
    nvtx.range_push(label)
    result = kernel.run_from_grouped_inputs(
        abc_tensors=abc,
        sfasfb_tensors=sfasfb,
        problem_sizes=problem_sizes,
        out_dtype=torch.bfloat16,
        alphas=alphas,
        alphas_packed=alphas_packed,
        iterations=0,
    )
    nvtx.range_pop()
    return result


# ── Phase 1: Compilation warmup ──────────────────────────────────────────────
nvtx.range_push("compile_scalar_alpha")
call_kernel("compile_scalar")
torch.cuda.synchronize()
nvtx.range_pop()

nvtx.range_push("compile_packed_alpha")
call_kernel("compile_packed", alphas_packed=_make_alphas_packed())
torch.cuda.synchronize()
nvtx.range_pop()

print("Compilation done. Starting profiled iterations...")
torch.cuda.synchronize()

# ── Phase 2: Profile 10 calls of scalar alpha (alphas=None) ──────────────────
nvtx.range_push("scalar_alpha_x10")
for i in range(10):
    call_kernel(f"scalar_{i}")
torch.cuda.synchronize()
nvtx.range_pop()

# ── Phase 3: Profile 10 calls of alphas_packed (per-token path) ──────────────
nvtx.range_push("packed_alpha_x10")
for i in range(10):
    call_kernel(f"packed_{i}", alphas_packed=_make_alphas_packed())
torch.cuda.synchronize()
nvtx.range_pop()

# ── Phase 4: Wall time comparison ────────────────────────────────────────────
def wall_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3

MEASURE = 50
print("\n── Steady-state wall time comparison ──")

t_none = sum(wall_ms(lambda: call_kernel("m")) for _ in range(MEASURE)) / MEASURE
print(f"  alphas=None    : {t_none*1e3:.1f} µs")

t_packed = sum(
    wall_ms(lambda: call_kernel("m", alphas_packed=_make_alphas_packed()))
    for _ in range(MEASURE)
) / MEASURE
print(f"  alphas_packed  : {t_packed*1e3:.1f} µs")

print("\nnsys profile complete. Check grouped_gemm_nsys.nsys-rep for GPU timeline.")

