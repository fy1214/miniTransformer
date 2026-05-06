"""
miniTransformer.distributed.cutedsl
====================================

基于 CUTLASS CuTeDSL + NVLS 的 tile-level GEMM + AllReduce 融合模块。

用法：
    from miniTransformer.distributed.cutedsl import GemmAllReduceCtx
    from miniTransformer.distributed.cutedsl import init_nvshmem_with_dist, finalize_nvshmem
"""

from .nvshmem_init import init_nvshmem_with_dist, finalize_nvshmem
from .gemm_allreduce import GemmAllReduceCtx

__all__ = [
    "init_nvshmem_with_dist",
    "finalize_nvshmem",
    "GemmAllReduceCtx",
]
