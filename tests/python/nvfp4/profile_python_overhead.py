"""
Fine-grained profiling of run_from_grouped_inputs Python-side overhead.

Instruments each logical stage with wall-clock timers to identify which
Python operations dominate the ~370 µs steady-state overhead.

Usage:
    python tests/python/nvfp4/profile_python_overhead.py
"""
import functools
import time
from collections import defaultdict
from typing import List, Optional
import torch

from miniTransformer.ops.cuteDSL.nvfp4.blockwise_group_gemm_cute_alpha_epilogue import (
    GroupedBlockScaledProfile,
    Sm100GroupedBlockScaledGemmKernel,
    _torch_dtype_to_cutlass,
    _resolve_profile,
    create_tensor_and_stride,
    generate_grouped_blockscaled_inputs,
)
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from miniTransformer.ops.cuteDSL.nvfp4.grouped_gemm_compile_cache import (
    build_grouped_compile_cache_key,
    get_or_compile_grouped_gemm,
)

DEFAULT_SHAPES = [
    (128, 4096, 7168),
    (128, 4096, 7168),
    (128, 4096, 7168),
    (128, 4096, 7168),
    (128, 4096, 7168),
    (128, 4096, 7168),
    (128, 4096, 7168),
    (128, 4096, 7168),
]

WARMUP = 10
MEASURE = 50


def now_us():
    return time.perf_counter() * 1e6


class StagedKernel(Sm100GroupedBlockScaledGemmKernel):
    """Subclass that replaces run_from_grouped_inputs with an instrumented version."""

    def run_instrumented(
        self,
        abc_tensors,
        sfasfb_tensors,
        problem_sizes,
        out_dtype: torch.dtype = torch.bfloat16,
        alphas: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        timings = {}

        # ------------------------------------------------------------------ #
        # Stage 1: alpha pack
        # ------------------------------------------------------------------ #
        device = abc_tensors["torch"][0][0].device
        t0 = now_us()
        if alphas is None:
            alphas = [
                torch.ones(m, dtype=torch.float32, device=device)
                for m, *_ in problem_sizes
            ]
        alpha_buf = Sm100GroupedBlockScaledGemmKernel._pack_alpha_buf(alphas, device)
        torch.cuda.synchronize()
        timings["1_pack_alpha"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 2: can_implement check
        # ------------------------------------------------------------------ #
        profile_cfg = self.profile_cfg
        c_dtype = _torch_dtype_to_cutlass(out_dtype)
        num_groups = len(problem_sizes)

        t0 = now_us()
        ok = Sm100GroupedBlockScaledGemmKernel.can_implement(
            profile_cfg.ab_dtype,
            profile_cfg.sf_dtype,
            profile_cfg.sf_vec_size,
            c_dtype,
            profile_cfg.mma_tiler_mn,
            profile_cfg.cluster_shape_mn,
            problem_sizes,
            profile_cfg.a_major,
            profile_cfg.b_major,
            profile_cfg.c_major,
        )
        timings["2_can_implement"] = now_us() - t0
        if not ok:
            raise TypeError("Unsupported testcase.")

        # ------------------------------------------------------------------ #
        # Stage 3: extract ptrs + strides (Python list traversal)
        # ------------------------------------------------------------------ #
        t0 = now_us()
        ptrs_abc = Sm100GroupedBlockScaledGemmKernel._extract_ptrs_from_groups(
            abc_tensors["torch"]
        )
        ptrs_sfasfb = Sm100GroupedBlockScaledGemmKernel._extract_ptrs_from_groups(
            sfasfb_tensors["torch"]
        )
        strides_abc = Sm100GroupedBlockScaledGemmKernel._infer_strides_from_abc_tensors(
            abc_tensors["torch"],
            profile_cfg.a_major,
            profile_cfg.b_major,
            profile_cfg.c_major,
        )
        timings["3_extract_ptrs_strides"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 4: alignment / divisibility constants (pure Python arithmetic)
        # ------------------------------------------------------------------ #
        t0 = now_us()
        alignment = 16
        divisibility_ab = 32 if profile_cfg.ab_dtype == cutlass.Float4E2M1FN else 16
        divisibility_c  = 32 if c_dtype == cutlass.Float4E2M1FN else 16
        divisibility_sf = 32 if profile_cfg.sf_dtype == cutlass.Float4E2M1FN else 16
        min_ab_size = alignment * 8 // profile_cfg.ab_dtype.width
        min_ab_size *= (divisibility_ab + min_ab_size - 1) // min_ab_size
        min_c_size = alignment * 8 // c_dtype.width
        min_c_size *= (divisibility_c + min_c_size - 1) // min_c_size
        min_sf_size = alignment * 8 // profile_cfg.sf_dtype.width
        min_sf_size *= (divisibility_sf + min_sf_size - 1) // min_sf_size
        timings["4_alignment_math"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 5: stub cache lookup / build
        # ------------------------------------------------------------------ #
        t0 = now_us()
        _stub_key = out_dtype
        if _stub_key not in self._stub_cache:
            self._stub_cache[_stub_key] = (
                [
                    create_tensor_and_stride(1, min_ab_size, min_ab_size, profile_cfg.a_major == "m", profile_cfg.ab_dtype)[2],
                    create_tensor_and_stride(1, min_ab_size, min_ab_size, profile_cfg.b_major == "n", profile_cfg.ab_dtype)[2],
                    create_tensor_and_stride(1, min_c_size,  min_c_size,  profile_cfg.c_major == "m", c_dtype)[2],
                ],
                [
                    create_tensor_and_stride(1, min_sf_size, min_sf_size, profile_cfg.a_major == "m", profile_cfg.sf_dtype)[2],
                    create_tensor_and_stride(1, min_sf_size, min_sf_size, profile_cfg.b_major == "n", profile_cfg.sf_dtype)[2],
                ],
            )
        initial_cute_tensors_abc, initial_cute_tensors_sfasfb = self._stub_cache[_stub_key]
        timings["5_stub_cache"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 6: hw cache + tensormap tensor lookup
        # ------------------------------------------------------------------ #
        t0 = now_us()
        cluster_n = profile_cfg.cluster_shape_mn[0] * profile_cfg.cluster_shape_mn[1]
        if 1 not in self._hw_cache or cluster_n not in self._hw_cache:
            _hw = cutlass.utils.HardwareInfo()
            self._hw_cache[1] = _hw.get_max_active_clusters(1)
            self._hw_cache[cluster_n] = (
                _hw.get_max_active_clusters(cluster_n) if cluster_n != 1 else self._hw_cache[1]
            )
        sm_count = self._hw_cache[1]
        max_active_clusters = self._hw_cache[cluster_n]
        tensormap_shape = (
            sm_count,
            Sm100GroupedBlockScaledGemmKernel.num_tensormaps,
            Sm100GroupedBlockScaledGemmKernel.bytes_per_tensormap // 8,
        )
        _tmaps = self._run_cache.setdefault("tmap", {})
        if sm_count not in _tmaps:
            _tmaps[sm_count], _ = cutlass_torch.cute_tensor_like(
                torch.empty(tensormap_shape, dtype=torch.int64, device=device),
                cutlass.Int64,
                is_dynamic_layout=False,
            )
        tensor_of_tensormap = _tmaps[sm_count]
        timings["6_hw_tmap_cache"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 7: shapes cache lookup (problem_sizes → cute tensors)
        # ------------------------------------------------------------------ #
        t0 = now_us()
        _shape_key = (tuple(tuple(ps) for ps in problem_sizes), str(out_dtype))
        _shapes = self._run_cache.setdefault("shapes", {})
        if _shape_key not in _shapes:
            if len(_shapes) >= self._SHAPES_CACHE_MAX:
                _shapes.pop(next(iter(_shapes)))
            _t_mnkl, _ = cutlass_torch.cute_tensor_like(
                torch.tensor(problem_sizes, dtype=torch.int32, device=device),
                cutlass.Int32, is_dynamic_layout=False, assumed_align=16,
            )
            _t_strides, _ = cutlass_torch.cute_tensor_like(
                torch.tensor(strides_abc, dtype=torch.int32, device=device),
                cutlass.Int32, is_dynamic_layout=False, assumed_align=16,
            )
            _shapes[_shape_key] = (_t_mnkl, _t_strides)
        tensor_of_dim_size_mnkl, tensor_of_strides_abc = _shapes[_shape_key]
        timings["7_shapes_cache"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 8: build tensor_of_ptrs_abc/sfasfb/alpha (torch.tensor + cute_tensor_like)
        # ------------------------------------------------------------------ #
        t0 = now_us()
        tensor_of_ptrs_abc, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(ptrs_abc, dtype=torch.int64, device=device),
            cutlass.Int64, is_dynamic_layout=False, assumed_align=16,
        )
        timings["8a_ptrs_abc_tensor"] = now_us() - t0

        t0 = now_us()
        tensor_of_ptrs_sfasfb, _ = cutlass_torch.cute_tensor_like(
            torch.tensor(ptrs_sfasfb, dtype=torch.int64, device=device),
            cutlass.Int64, is_dynamic_layout=False, assumed_align=16,
        )
        timings["8b_ptrs_sfasfb_tensor"] = now_us() - t0

        t0 = now_us()
        tensor_of_alpha_buf, _ = cutlass_torch.cute_tensor_like(
            alpha_buf, cutlass.Float32, is_dynamic_layout=False, assumed_align=16,
        )
        timings["8c_alpha_buf_tensor"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 9: compute_total_num_clusters
        # ------------------------------------------------------------------ #
        t0 = now_us()
        cluster_tile_shape_mn = (
            128 * profile_cfg.cluster_shape_mn[0],
            profile_cfg.mma_tiler_mn[1] * profile_cfg.cluster_shape_mn[1],
        )
        total_num_clusters = sum(
            functools.reduce(lambda x, y: x * y,
                             ((x + y - 1) // y for x, y in zip((m, n), cluster_tile_shape_mn)))
            for m, n, _, _ in problem_sizes
        )
        timings["9_total_clusters"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 10: compile cache key build + lookup
        # ------------------------------------------------------------------ #
        t0 = now_us()
        profile_name = self.profile.value if self.profile is not None else "custom"
        compile_cache_key = build_grouped_compile_cache_key(
            variant="alpha_epilogue",
            device=device,
            profile_name=profile_name,
            profile_cfg=profile_cfg,
            num_groups=num_groups,
            problem_sizes=problem_sizes,
            out_dtype=out_dtype,
            host_problem_shape_available=True,
        )
        compiled_grouped_gemm = get_or_compile_grouped_gemm(
            compile_cache_key, lambda: None  # should always hit cache here
        )
        timings["10_compile_cache"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 10b: default_stream()
        # ------------------------------------------------------------------ #
        t0 = now_us()
        current_stream = cutlass_torch.default_stream()
        timings["10b_default_stream"] = now_us() - t0

        # ------------------------------------------------------------------ #
        # Stage 11: kernel dispatch (async, no sync — mirrors production path)
        # ------------------------------------------------------------------ #
        t0 = now_us()
        compiled_grouped_gemm(
            initial_cute_tensors_abc[0],
            initial_cute_tensors_abc[1],
            initial_cute_tensors_abc[2],
            initial_cute_tensors_sfasfb[0],
            initial_cute_tensors_sfasfb[1],
            tensor_of_dim_size_mnkl,
            tensor_of_strides_abc,
            tensor_of_ptrs_abc,
            tensor_of_ptrs_sfasfb,
            tensor_of_alpha_buf,
            tensor_of_tensormap,
            current_stream,
        )
        timings["11_dispatch_async"] = now_us() - t0

        return timings


def main():
    profile = GroupedBlockScaledProfile["NVFP4"]
    shapes = DEFAULT_SHAPES
    m_list = [s[0] for s in shapes]
    n_list = [s[1] for s in shapes]
    k_list = [s[2] for s in shapes]

    abc, sfasfb, problem_sizes = generate_grouped_blockscaled_inputs(
        m_list=m_list, n_list=n_list, k_list=k_list,
        profile=profile, device="cuda", out_dtype=torch.bfloat16,
    )

    kernel = StagedKernel(profile=profile)

    # Trigger compilation via the normal path first
    print("Compiling kernel (first call)...")
    kernel.run_from_grouped_inputs(
        abc_tensors=abc, sfasfb_tensors=sfasfb,
        problem_sizes=problem_sizes, out_dtype=torch.bfloat16,
    )
    print("Done. Warming up instrumented path...")

    def prod_call():
        kernel.run_from_grouped_inputs(
            abc_tensors=abc, sfasfb_tensors=sfasfb,
            problem_sizes=problem_sizes, out_dtype=torch.bfloat16,
            iterations=0,
        )

    # Warmup
    for _ in range(WARMUP):
        kernel.run_instrumented(abc, sfasfb, problem_sizes)
        prod_call()

    # Ground-truth: wall time of real run_from_grouped_inputs with iterations=0 + sync
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        prod_call()
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - t0) / MEASURE * 1e6

    # Measure per-stage
    accum = defaultdict(float)
    for _ in range(MEASURE):
        t = kernel.run_instrumented(abc, sfasfb, problem_sizes)
        for k, v in t.items():
            accum[k] += v

    total_tracked = sum(accum.values()) / MEASURE
    untracked_us = wall_us - total_tracked

    print(f"\nGround-truth wall time (run_from_grouped_inputs): {wall_us:.1f} µs")
    print(f"\nPer-stage breakdown  (mean over {MEASURE} iters)")
    print(f"{'Stage':<38}  {'µs':>8}  {'%wall':>7}")
    print("-" * 60)
    for stage, total in sorted(accum.items()):
        us = total / MEASURE
        pct = us / wall_us * 100
        print(f"  {stage:<36}  {us:>8.1f}  {pct:>6.1f}%")
    print("-" * 60)
    print(f"  {'tracked subtotal':<36}  {total_tracked:>8.1f}  {total_tracked/wall_us*100:>6.1f}%")
    print(f"  {'untracked (gaps between stages)':<36}  {untracked_us:>8.1f}  {untracked_us/wall_us*100:>6.1f}%")
    print(f"  {'TOTAL (wall)':<36}  {wall_us:>8.1f}  {'100.0':>6}%")


if __name__ == "__main__":
    main()
