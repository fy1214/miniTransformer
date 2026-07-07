"""
gemm_allreduce.py
=================

row parallel GEMM + tile-level fp32 AllReduce 融合（基于 CUTLASS CuTeDSL + NVLS）。

每个 CTA 计算完自己的 C[mi, ni] tile 后，立即通过 multimem.ld_reduce + multimem.st
（NVLS 硬件加速）对该 tile 做跨 rank AllReduce，无需等整个 GEMM 完成。

完全独立：不依赖 miniTransformer 其他模块，输入/输出是标准 torch.Tensor。

用法：
    # 1. 进程启动时，nvshmem 初始化（一次）
    from miniTransformer.distributed.cutedsl.nvshmem_init import init_nvshmem_with_dist
    dev = init_nvshmem_with_dist()

    # 2. 每个层初始化一次 ctx
    ctx = GemmAllReduceCtx(tp_group=tp_group)

    # 3. 每次 forward（同步由外部控制）
    y = ctx.forward(a, b)
    # a: [M, K] bfloat16，row-major
    # b: [N, K] bfloat16，row-major（GEMM 计算 a @ b.T）
    # y: [M, N] bfloat16，已完成 AllReduce

    # 若会继续复用同一个 ctx 做下一次 forward，建议在外层执行：
    #   ctx.barrier_before_reuse()
    #
    # 4. 进程退出前释放
    ctx.free()
    from miniTransformer.distributed.cutedsl.nvshmem_init import finalize_nvshmem
    finalize_nvshmem()

注意：
  - 需要先调用 init_nvshmem_with_dist() 才能使用此类
  - nvshmem 对称内存需要手动调用 ctx.free() 释放
  - 目前仅支持 BFloat16 输入，Float32 累加，BFloat16 输出（row parallel 典型场景）
"""

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

try:
    import nvshmem.core
except ImportError:
    raise ImportError("nvshmem4py required. Run init_nvshmem_with_dist() first.")

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import BFloat16

try:
    import cuda.bindings.driver as cuda_driver
    from cuda.core.experimental import Device
except ImportError:
    import cuda.bindings.driver as cuda_driver
    from cuda.core import Device

from .kernel import PersistentDenseGemmKernel, create_mc_tensor


# ── 默认 kernel 配置（bf16 row-parallel 场景）─────────────────────────────────
# ncu 结果：SM 76%，MMA 76%，long_scoreboard stall 87%，warps_active 7%
# 当前瓶颈是 dist.barrier() 同步开销，不是 tile 大小
# 256×128 + 2-CTA 在小矩阵下反而因 cluster 同步开销更慢，保持 128×128
_DEFAULT_MMA_TILER_MN    = (128, 128)
_DEFAULT_CLUSTER_SHAPE_MN = (1, 1)
_DEFAULT_USE_2CTA        = False
_DEFAULT_USE_TMA_STORE   = False    # non-TMA path（更简单，已验证）
_DEFAULT_ALL_REDUCE      = "LDMCxSTMC"


def _get_max_active_clusters():
    """获取当前 GPU 支持的最大活跃 cluster 数。"""
    props = torch.cuda.get_device_properties("cuda")
    return props.multi_processor_count


class GemmAllReduceCtx:
    """
    row parallel GEMM + tile-level fp32 AllReduce。

    Lazy compile：第一次 forward 时根据实际 tensor shape 和 dtype JIT 编译 kernel。
    nvshmem 对称内存在 _compile() 时分配，在 free() 时释放。

    同步策略：
      - forward() 内不做 torch/distributed 同步，保持 kernel 路径纯计算。
      - 跨 rank 时序由外层控制；若会复用同一个 ctx 进入下一轮 forward，
        建议在两轮之间调用 barrier_before_reuse()。
    """

    def __init__(
        self,
        tp_group: Optional[dist.ProcessGroup] = None,
        mma_tiler_mn: Tuple[int, int] = _DEFAULT_MMA_TILER_MN,
        cluster_shape_mn: Tuple[int, int] = _DEFAULT_CLUSTER_SHAPE_MN,
        use_2cta_instrs: bool = _DEFAULT_USE_2CTA,
        use_tma_store: bool = _DEFAULT_USE_TMA_STORE,
        all_reduce: str = _DEFAULT_ALL_REDUCE,
    ):
        """
        Args:
            tp_group: TP 进程组，None 使用 default group
            mma_tiler_mn: MMA tile 的 (M, N) 形状，编译时固化
            cluster_shape_mn: cluster 的 (M, N) 形状，编译时固化
            use_2cta_instrs: 是否使用 2-CTA MMA 指令
            use_tma_store: 是否使用 TMA store 写出 C（False 更简单，已验证）
            all_reduce: AllReduce 模式，"LDMCxSTMC" 使用 NVLS 硬件加速
        """
        self._tp_group       = tp_group
        self._rank           = dist.get_rank(tp_group) if dist.is_initialized() else 0
        self._world_size     = dist.get_world_size(tp_group) if dist.is_initialized() else 1
        self._mma_tiler_mn   = mma_tiler_mn
        self._cluster_shape  = cluster_shape_mn
        self._use_2cta       = use_2cta_instrs
        self._use_tma_store  = use_tma_store
        self._all_reduce     = all_reduce

        # lazy-initialized on first forward
        self._compiled        = None
        self._gemm_kernel     = None

        # nvshmem symmetric tensors（torch handle，手动管理生命周期）
        self._c_torch         = None
        self._c_mc_torch      = None
        self._flag_torch      = None
        self._flag_mc_torch   = None

        # CuTe tensor handles（传给 compiled kernel）
        self._c_cute          = None
        self._c_mc_cute       = None
        self._flag_cute       = None
        self._flag_mc_cute    = None

        self._M = None
        self._N = None
        self._K = None

    def _compile(self, a: torch.Tensor, b: torch.Tensor):
        """
        根据第一次 forward 的 tensor shape/dtype JIT 编译 kernel。

        a: [M, K] bfloat16
        b: [N, K] bfloat16（row-major，GEMM 计算 a @ b.T）
        """
        assert a.dtype == torch.bfloat16, f"Expected bfloat16, got {a.dtype}"
        assert b.dtype == torch.bfloat16, f"Expected bfloat16, got {b.dtype}"
        assert a.is_cuda and b.is_cuda, "Tensors must be on CUDA"
        assert a.is_contiguous() and b.is_contiguous(), "Tensors must be contiguous"

        M, K = a.shape
        N    = b.shape[0]
        self._M, self._N, self._K = M, N, K

        # A/B 使用 BFloat16（保持 bf16 GEMM 吞吐）
        # C 使用 Float32（让 tile-level AllReduce 在 fp32 精度下进行）
        # 最后 cast fp32→bf16 输出
        ab_dtype = cutlass.BFloat16
        c_dtype  = cutlass.Float32

        num_tiles = (
            (M // self._mma_tiler_mn[0]) *
            (N // self._mma_tiler_mn[1])
        )
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

        # ── Step 1: 分配 nvshmem 对称内存给 C 和 barrier_flag ─────────────────
        # C 需要 [M, N, L=1] 形状（kernel 内部使用 3D tensor），dtype Float32
        # 最后返回时 cast 回 bf16
        c_cpu   = torch.zeros(M, N, 1, dtype=torch.float32)   # CPU tensor，nvshmem 会 copy 到 GPU
        # barrier_flag: [num_tiles + num_sms] int32，必须是 CPU tensor
        flag_cpu = torch.zeros(num_tiles + num_sms, dtype=torch.int32)  # CPU，不能用 device="cuda"

        # create_mc_tensor: nvshmem.core.tensor + multicast view
        # leading_dim=1 表示 row-major C（N 是 leading dimension，col-major 用 0）
        self._c_cute, self._c_mc_cute, self._c_torch, self._c_mc_torch = create_mc_tensor(
            c_cpu, c_dtype, leading_dim=1, is_dynamic_layout=True
        )
        self._flag_cute, self._flag_mc_cute, self._flag_torch, self._flag_mc_torch = create_mc_tensor(
            flag_cpu, cutlass.Int32, leading_dim=0, is_dynamic_layout=True
        )

        # ── Step 2: 构造 A/B 的 CuTe tensor 模板（用于 compile）─────────────
        # A row-major [M, K]: a_major="k" → is_M_major=False
        # B row-major [N, K]: b_major="k" → is_N_major=False
        # A/B 用 BFloat16：保持 bf16 GEMM 吞吐
        # C 用 Float32：fp32 AllReduce 精度，最后 cast 回 bf16
        a_cpu_template = cutlass_torch.matrix(1, M, K, False, ab_dtype)  # [M, K, 1] bf16
        b_cpu_template = cutlass_torch.matrix(1, N, K, False, ab_dtype)  # [N, K, 1] bf16

        a_cute, _ = cutlass_torch.cute_tensor_like(
            a_cpu_template, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )
        b_cute, _ = cutlass_torch.cute_tensor_like(
            b_cpu_template, ab_dtype, is_dynamic_layout=True, assumed_align=16
        )

        # ── Step 3: 构造 kernel 实例 ────────────────────────────────────────
        self._gemm_kernel = PersistentDenseGemmKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=self._use_2cta,
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape,
            use_tma_store=self._use_tma_store,
            rank_id=self._rank,
            world_size=self._world_size,
            all_reduce=self._all_reduce,
        )

        # ── Step 4: JIT 编译（固化 shape/dtype，留 stride 动态）─────────────
        max_clusters = _get_max_active_clusters()
        torch_stream = torch.cuda.current_stream()
        current_stream = cuda_driver.CUstream(torch_stream.cuda_stream)

        self._compiled = cute.compile(
            self._gemm_kernel,
            a_cute,
            b_cute,
            self._c_cute,
            self._c_mc_cute,
            self._flag_cute,
            self._flag_mc_cute,
            max_clusters,
            current_stream,
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        执行 row parallel GEMM + tile-level AllReduce。

        Args:
            a: [M, K] bfloat16，row-major，contiguous
            b: [N, K] bfloat16，row-major，contiguous
               （GEMM 计算 a @ b.T，b 存储方式等价于 B 矩阵的转置）

        Returns:
            [M, N] bfloat16，已完成 fp32 tile-level AllReduce
        """
        a = a.contiguous()
        b = b.contiguous()

        # lazy compile
        if self._compiled is None:
            self._compile(a, b)

        # 清零 flag 和 C
        self._flag_torch.zero_()
        self._c_torch.zero_()

        # A/B 保持 bf16，unsqueeze 加 L 维度后构造 cute tensor
        ab_dtype = cutlass.BFloat16
        a_3d = a.unsqueeze(-1).contiguous()   # bf16 [M, K, 1]
        b_3d = b.unsqueeze(-1).contiguous()   # bf16 [N, K, 1]

        a_cute = from_dlpack(a_3d, assumed_align=16)
        a_cute.element_type = ab_dtype
        a_cute = a_cute.mark_layout_dynamic(leading_dim=1)

        b_cute = from_dlpack(b_3d, assumed_align=16)
        b_cute.element_type = ab_dtype
        b_cute = b_cute.mark_layout_dynamic(leading_dim=1)

        torch_stream = torch.cuda.current_stream()
        current_stream = cuda_driver.CUstream(torch_stream.cuda_stream)

        # 调用编译好的 kernel
        self._compiled(
            a_cute,
            b_cute,
            self._c_cute,
            self._c_mc_cute,
            self._flag_cute,
            self._flag_mc_cute,
            current_stream,
        )

        # 等待 kernel 完成，返回结果
        # _c_torch 是 fp32 [M, N, 1]，cast 回 bf16 并 squeeze L 维度
        # torch.cuda.synchronize()
        return self._c_torch.squeeze(-1).to(torch.bfloat16).clone()

    def barrier_before_reuse(self):
        """
        在复用同一个 ctx 发起下一次 forward 前做跨 rank 同步。

        该同步是”外层显式控制”接口，forward() 本身不会触发同步。
        """
        if dist.is_initialized() and self._world_size > 1:
            torch.cuda.synchronize()
            dist.barrier(group=self._tp_group)

    def free(self):
        """
        手动释放 nvshmem 对称内存。
        必须在 finalize_nvshmem() 之前调用。
        """
        for t in [self._c_torch, self._c_mc_torch,
                  self._flag_torch, self._flag_mc_torch]:
            if t is not None:
                try:
                    nvshmem.core.free_tensor(t)
                except Exception:
                    pass

        self._c_torch = self._c_mc_torch = None
        self._flag_torch = self._flag_mc_torch = None
        self._c_cute = self._c_mc_cute = None
        self._flag_cute = self._flag_mc_cute = None
        self._compiled = None
        self._gemm_kernel = None

    def __del__(self):
        # 尽力释放，忽略错误（进程退出时可能 nvshmem 已 finalize）
        try:
            self.free()
        except Exception:
            pass
