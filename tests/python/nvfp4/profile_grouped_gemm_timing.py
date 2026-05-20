"""
Profile Sm100GroupedBlockScaledGemmKernel to distinguish compilation time vs kernel execution time.

Usage:
    python profile_grouped_gemm_timing.py
    python profile_grouped_gemm_timing.py --profile MXFP4
    python profile_grouped_gemm_timing.py --no-disk-cache
"""
import argparse
import os
import time

import torch

from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute_alpha_epilogue import (
    GroupedBlockScaledProfile,
    Sm100GroupedBlockScaledGemmKernel,
    generate_grouped_blockscaled_inputs,
)

# ---------------------------------------------------------------------------
# Test shapes: MoE-style, 8 experts
# ---------------------------------------------------------------------------
# Modify to match your actual workload
DEFAULT_SHAPES = [
    # (M,   N,    K)
    (128,  4096, 7168),
    (128,  4096, 7168),
    (128,  4096, 7168),
    (128,  4096, 7168),
    (128,  4096, 7168),
    (128,  4096, 7168),
    (128,  4096, 7168),
    (128,  4096, 7168),
]

WARMUP_ITERS = 5
BENCH_ITERS  = 20


def wall_time_ms(fn):
    """Measure wall-clock time of fn() in milliseconds."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def cuda_time_ms(fn, warmup=0, iters=1):
    """Measure CUDA event time of fn() in milliseconds (mean over iters).

    WARNING: if fn() contains Python host work before kernel launch, the CUDA
    event will also capture GPU-idle time waiting for the host.  Use
    kernel_time_ms() instead when fn() wraps run_from_grouped_inputs().
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def kernel_time_ms(kernel, abc, sfasfb, problem_sizes, out_dtype,
                   alphas=None, warmup=5, iters=20):
    """Measure pure GPU kernel time using the built-in iterations loop.

    run_from_grouped_inputs(iterations=N) runs N kernel launches inside a
    single Python call, timing them with CUDA events at the C++/CUDA layer.
    Python dispatch overhead is paid only once, not per iteration.
    Returns average per-kernel time in milliseconds.
    """
    kernel.run_from_grouped_inputs(
        abc_tensors=abc, sfasfb_tensors=sfasfb,
        problem_sizes=problem_sizes,
        out_dtype=out_dtype,
        alphas=alphas,
        warmup_iterations=warmup,
        iterations=iters,
    )
    metrics = kernel.last_benchmark_metrics
    if not metrics:
        return float("nan")
    # metrics has one entry per group; avg_runtime_us covers all groups together
    return metrics[0]["avg_runtime_us"] / 1e3


def tflops(problem_sizes, elapsed_ms):
    flops = sum(2 * m * n * k for m, n, k, _ in problem_sizes)
    return flops / (elapsed_ms * 1e-3) / 1e12


def sep(char="=", width=72):
    print(char * width)


def run_profile(profile_name: str, use_disk_cache: bool, shapes=None):
    if shapes is None:
        shapes = DEFAULT_SHAPES

    profile = GroupedBlockScaledProfile[profile_name]
    m_list  = [s[0] for s in shapes]
    n_list  = [s[1] for s in shapes]
    k_list  = [s[2] for s in shapes]

    sep()
    print(f"Profile : {profile_name}")
    print(f"Groups  : {len(shapes)}")
    print(f"Shapes  : M={m_list[0]} N={n_list[0]} K={k_list[0]} (all groups identical)")
    print(f"Disk cache : {'enabled' if use_disk_cache else 'disabled'}")
    sep()

    # ------------------------------------------------------------------
    # 1. Generate inputs (always fresh, not counted in kernel timing)
    # ------------------------------------------------------------------
    def make_inputs():
        return generate_grouped_blockscaled_inputs(
            m_list=m_list, n_list=n_list, k_list=k_list,
            profile=profile, device="cuda", out_dtype=torch.bfloat16,
        )

    abc, sfasfb, problem_sizes = make_inputs()

    # ------------------------------------------------------------------
    # 2. First call — includes JIT compilation
    # ------------------------------------------------------------------
    kernel = Sm100GroupedBlockScaledGemmKernel(profile=profile)

    def call_kernel(abc_, sfasfb_):
        return kernel.run_from_grouped_inputs(
            abc_tensors=abc_, sfasfb_tensors=sfasfb_,
            problem_sizes=problem_sizes,
            out_dtype=torch.bfloat16,
        )

    print("\n[1] First call (compilation + single execution)")
    t_first = wall_time_ms(lambda: call_kernel(abc, sfasfb))
    print(f"    Wall time : {t_first:.1f} ms")

    # ------------------------------------------------------------------
    # 3. Second call — same shapes, should hit in-memory cache
    # ------------------------------------------------------------------
    print("\n[2] Second call (cache hit, single execution)")
    t_second = wall_time_ms(lambda: call_kernel(abc, sfasfb))
    print(f"    Wall time : {t_second:.1f} ms")
    print(f"    Compilation overhead : ~{t_first - t_second:.1f} ms")

    # ------------------------------------------------------------------
    # 3. Pure GPU kernel time (built-in iterations loop, correct method)
    #    Python dispatch overhead is paid only once for N iterations.
    # ------------------------------------------------------------------
    print(f"\n[3] GPU kernel time (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    t_exec = kernel_time_ms(kernel, abc, sfasfb, problem_sizes, torch.bfloat16,
                            warmup=WARMUP_ITERS, iters=BENCH_ITERS)
    tflops_val = tflops(problem_sizes, t_exec)
    print(f"    Kernel mean : {t_exec:.3f} ms  ({t_exec*1e3:.1f} µs)")
    print(f"    TFLOPS      : {tflops_val:.2f}")

    # ------------------------------------------------------------------
    # 4. Steady-state wall time (full call, all caches warm)
    # ------------------------------------------------------------------
    STEADY_WARMUP  = 10
    STEADY_MEASURE = 50

    def full_call():
        kernel.run_from_grouped_inputs(
            abc_tensors=abc, sfasfb_tensors=sfasfb,
            problem_sizes=problem_sizes,
            out_dtype=torch.bfloat16,
        )

    for _ in range(STEADY_WARMUP):
        full_call()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(STEADY_MEASURE):
        full_call()
    torch.cuda.synchronize()
    t_steady_us = (time.perf_counter() - t0) / STEADY_MEASURE * 1e6

    print(f"\n[4] Steady-state full call (warmup={STEADY_WARMUP}, iters={STEADY_MEASURE})")
    print(f"    Wall time (mean) : {t_steady_us:.1f} µs")
    print(f"    GPU kernel       : {t_exec*1e3:.1f} µs")
    print(f"    Python overhead  : ~{t_steady_us - t_exec*1e3:.1f} µs")

    sep()
    print("Summary")
    sep("-")
    print(f"  Compilation          : ~{t_first - t_second:.0f} ms  (one-time per shape)")
    print(f"  GPU kernel           : {t_exec*1e3:.1f} µs  @ {tflops_val:.2f} TFLOPS")
    print(f"  Steady-state (full)  : {t_steady_us:.1f} µs")
    print(f"  Python overhead      : ~{t_steady_us - t_exec*1e3:.1f} µs")
    sep()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="NVFP4",
                        choices=["NVFP4", "MXFP4", "MXFP8"])
    parser.add_argument("--no-disk-cache", action="store_true")
    args = parser.parse_args()

    if args.no_disk_cache:
        os.environ.pop("NVFP4_CUTE_PERSISTENT_CACHE", None)
    else:
        os.environ.setdefault("NVFP4_CUTE_PERSISTENT_CACHE", "1")

    run_profile(args.profile, use_disk_cache=not args.no_disk_cache)


if __name__ == "__main__":
    main()
