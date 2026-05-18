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
    """Measure CUDA event time of fn() in milliseconds (mean over iters)."""
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
    # 4. Pure kernel execution time via CUDA events (with warmup)
    # ------------------------------------------------------------------
    print(f"\n[3] Kernel execution time (warmup={WARMUP_ITERS}, iters={BENCH_ITERS})")
    t_exec = cuda_time_ms(
        lambda: call_kernel(abc, sfasfb),
        warmup=WARMUP_ITERS,
        iters=BENCH_ITERS,
    )
    tflops_val = tflops(problem_sizes, t_exec)
    print(f"    CUDA event mean : {t_exec:.3f} ms")
    print(f"    TFLOPS          : {tflops_val:.2f}")

    # ------------------------------------------------------------------
    # 5. Per-token alpha path (different epilogue => separate compilation)
    # ------------------------------------------------------------------
    print("\n[4] Per-token alpha path (separate compilation)")
    alphas_pertoken = [
        torch.ones(m, device="cuda", dtype=torch.float32) for m in m_list
    ]

    def call_kernel_pertoken(abc_, sfasfb_):
        return kernel.run_from_grouped_inputs(
            abc_tensors=abc_, sfasfb_tensors=sfasfb_,
            problem_sizes=problem_sizes,
            out_dtype=torch.bfloat16,
            alphas=alphas_pertoken,
        )

    t_pt_first = wall_time_ms(lambda: call_kernel_pertoken(abc, sfasfb))
    print(f"    First call  : {t_pt_first:.1f} ms  (includes per-token epilogue compilation)")

    t_pt_exec = cuda_time_ms(
        lambda: call_kernel_pertoken(abc, sfasfb),
        warmup=WARMUP_ITERS,
        iters=BENCH_ITERS,
    )
    print(f"    Kernel mean : {t_pt_exec:.3f} ms")

    # ------------------------------------------------------------------
    # 6. Sweep M sizes to show per-shape compilation cost
    # ------------------------------------------------------------------
    print("\n[5] Compilation cost sweep — varying M per group")
    sep("-")
    print(f"  {'M':>6}  {'1st call (ms)':>14}  {'2nd call (ms)':>14}  {'exec (ms)':>10}  {'TFLOPS':>8}")
    sep("-")

    sweep_kernel = Sm100GroupedBlockScaledGemmKernel(profile=profile)
    for m in [32, 64, 128, 256, 512]:
        ml = [m] * len(shapes)
        try:
            abc_s, sfasfb_s, ps_s = generate_grouped_blockscaled_inputs(
                m_list=ml, n_list=n_list, k_list=k_list,
                profile=profile, device="cuda", out_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"  {m:>6}  skipped ({e})")
            continue

        def _call(a=abc_s, s=sfasfb_s, p=ps_s):
            return sweep_kernel.run_from_grouped_inputs(
                abc_tensors=a, sfasfb_tensors=s,
                problem_sizes=p, out_dtype=torch.bfloat16,
            )

        t1 = wall_time_ms(_call)
        t2 = wall_time_ms(_call)
        te = cuda_time_ms(_call, warmup=3, iters=10)
        tf = tflops(ps_s, te)
        print(f"  {m:>6}  {t1:>14.1f}  {t2:>14.1f}  {te:>10.3f}  {tf:>8.2f}")

    sep()
    print("Summary")
    sep("-")
    print(f"  Compilation time (scalar alpha)     : ~{t_first - t_second:.0f} ms")
    print(f"  Compilation time (pertoken alpha)   : ~{t_pt_first - t_second:.0f} ms  (incremental)")
    print(f"  Kernel execution (scalar alpha)     : {t_exec:.3f} ms  @ {tflops_val:.2f} TFLOPS")
    print(f"  Kernel execution (pertoken alpha)   : {t_pt_exec:.3f} ms")
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
