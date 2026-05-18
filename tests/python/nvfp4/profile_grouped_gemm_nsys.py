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

# ── Pertoken alpha (the slow path under investigation) ──────────────────────
alphas_pertoken = [torch.ones(M, device="cuda", dtype=torch.float32) for _ in range(G)]
alphas_scalar   = None

# ── Generate inputs once (not part of kernel timing) ─────────────────────────
nvtx.range_push("generate_inputs")
abc, sfasfb, problem_sizes = generate_grouped_blockscaled_inputs(
    m_list=m_list, n_list=n_list, k_list=k_list,
    profile=PROFILE, device="cuda", out_dtype=torch.bfloat16,
)
nvtx.range_pop()

kernel = Sm100GroupedBlockScaledGemmKernel(profile=PROFILE)

# ── Helper ───────────────────────────────────────────────────────────────────
def call_kernel(alphas, label):
    nvtx.range_push(label)
    result = kernel.run_from_grouped_inputs(
        abc_tensors=abc,
        sfasfb_tensors=sfasfb,
        problem_sizes=problem_sizes,
        out_dtype=torch.bfloat16,
        alphas=alphas,
        # iterations=0: skip benchmark block → no synchronize() inside
        # This lets CUDA events in the caller measure pure GPU time
        iterations=0,
    )
    nvtx.range_pop()
    return result


# ── Phase 1: Compilation warmup (not the focus, just trigger compile) ────────
nvtx.range_push("compile_scalar_alpha")
call_kernel(alphas_scalar, "compile_scalar")
torch.cuda.synchronize()
nvtx.range_pop()

nvtx.range_push("compile_pertoken_alpha")
call_kernel(alphas_pertoken, "compile_pertoken")
torch.cuda.synchronize()
nvtx.range_pop()

print("Compilation done. Starting profiled iterations...")
torch.cuda.synchronize()

# ── Phase 2: Profile 10 calls of SCALAR alpha ─────────────────────────────
# These should be fast if Python dispatch overhead is the issue.
nvtx.range_push("scalar_alpha_x10")
for i in range(10):
    call_kernel(alphas_scalar, f"scalar_{i}")
torch.cuda.synchronize()
nvtx.range_pop()

# ── Phase 3: Profile 10 calls of PERTOKEN alpha ───────────────────────────
# Compare GPU idle time vs scalar path above.
nvtx.range_push("pertoken_alpha_x10")
for i in range(10):
    call_kernel(alphas_pertoken, f"pertoken_{i}")
torch.cuda.synchronize()
nvtx.range_pop()

# ── Phase 4: Isolate alpha descriptor packing cost ───────────────────────────
# Call _pack_alpha_descriptors directly to measure its cost.
nvtx.range_push("pack_alpha_descriptors_x100")
for _ in range(100):
    nvtx.range_push("pack_alpha_scalar")
    Sm100GroupedBlockScaledGemmKernel._pack_alpha_descriptors(
        alphas_scalar, problem_sizes, torch.device("cuda")
    )
    nvtx.range_pop()
    nvtx.range_push("pack_alpha_pertoken")
    Sm100GroupedBlockScaledGemmKernel._pack_alpha_descriptors(
        alphas_pertoken, problem_sizes, torch.device("cuda")
    )
    nvtx.range_pop()
torch.cuda.synchronize()
nvtx.range_pop()

# ── Phase 5: Measure each sub-phase with wall time ───────────────────────────
# This runs on CPU only (no nsys needed) but helps narrow down the bottleneck.
import time

def wall_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3

print("\n── Sub-phase wall time breakdown (scalar alpha) ──")

import cutlass
import cutlass_torch
from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute_alpha_epilogue import (
    create_tensor_and_stride,
    create_tensors_sfasfb_for_all_groups,
)
from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_compile_cache import (
    build_grouped_compile_cache_key,
    get_or_compile_grouped_gemm,
)

profile_cfg = kernel.profile_cfg

# (a) _pack_alpha_descriptors
t = wall_ms(lambda: Sm100GroupedBlockScaledGemmKernel._pack_alpha_descriptors(
    alphas_scalar, problem_sizes, torch.device("cuda")
))
print(f"  _pack_alpha_descriptors (scalar)  : {t:.3f} ms")

t = wall_ms(lambda: Sm100GroupedBlockScaledGemmKernel._pack_alpha_descriptors(
    alphas_pertoken, problem_sizes, torch.device("cuda")
))
print(f"  _pack_alpha_descriptors (pertoken): {t:.3f} ms")

# (b) create_tensor_and_stride for dummy tensors (x5: A, B, C, SFA, SFB)
min_ab_size  = max(profile_cfg.sf_vec_size * 2, 32)
min_sf_size  = 32
min_c_size   = 32
a_dtype  = profile_cfg.ab_dtype
sf_dtype = profile_cfg.sf_dtype
t = wall_ms(lambda: [
    create_tensor_and_stride(1, min_ab_size, min_ab_size, a_dtype,  True,  "cuda"),
    create_tensor_and_stride(1, min_ab_size, min_ab_size, a_dtype,  True,  "cuda"),
    create_tensor_and_stride(1, min_c_size,  min_c_size,  torch.bfloat16, True,  "cuda"),
    create_tensor_and_stride(1, min_sf_size, min_sf_size, sf_dtype, False, "cuda"),
    create_tensor_and_stride(1, min_sf_size, min_sf_size, sf_dtype, False, "cuda"),
])
print(f"  create_tensor_and_stride ×5       : {t:.3f} ms")

# (c) HardwareInfo query
t = wall_ms(lambda: cutlass.utils.HardwareInfo())
print(f"  HardwareInfo()                    : {t:.3f} ms")

# (d) torch.tensor H2D copies (ptrs + strides + problem_sizes)
ptrs_abc    = Sm100GroupedBlockScaledGemmKernel._extract_ptrs_from_groups(abc["torch"])
ptrs_sfasfb = Sm100GroupedBlockScaledGemmKernel._extract_ptrs_from_groups(sfasfb["torch"])
strides_abc = Sm100GroupedBlockScaledGemmKernel._infer_strides_from_abc_tensors(
    abc["torch"], profile_cfg.a_major, profile_cfg.b_major, profile_cfg.c_major,
)
t = wall_ms(lambda: [
    torch.tensor(problem_sizes,  dtype=torch.int32,  device="cuda"),
    torch.tensor(strides_abc,    dtype=torch.int64,  device="cuda"),
    torch.tensor(ptrs_abc,       dtype=torch.int64,  device="cuda"),
    torch.tensor(ptrs_sfasfb,    dtype=torch.int64,  device="cuda"),
])
print(f"  torch.tensor H2D copies ×4        : {t:.3f} ms")

# (e) torch.cuda.synchronize cost
t = wall_ms(torch.cuda.synchronize)
print(f"  torch.cuda.synchronize()          : {t:.3f} ms")

print("\nnsys profile complete. Check grouped_gemm_nsys.nsys-rep for GPU timeline.")
print("Key question: in the nsys timeline, is the GPU kernel duration < 5ms?")
print("If yes, the 240ms is all Python host overhead.")
