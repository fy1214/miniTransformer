import argparse
import fcntl
import hashlib
import os
import pickle
import sys
import tempfile
from dataclasses import dataclass
from getpass import getuser
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.torch as cutlass_torch
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import from_dlpack, make_ptr

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(current_dir, "..", "..")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

mma_tiler_mn = (256, 256)
mma_inst_shape_k = 64
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN

sf_vec_size = 16
cluster_shape_mnk = (2, 1, 1)

DEFAULT_MMA_TILER_MN = mma_tiler_mn
DEFAULT_CLUSTER_SHAPE_MNK = cluster_shape_mnk
DEFAULT_NUM_AB_STAGE = 5
DEFAULT_NUM_ACC_STAGE = 1


@dataclass(frozen=True)
class GemmTuneConfig:
    mma_tiler_mn: Tuple[int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    use_2cta_instrs: bool
    num_ab_stage: int
    num_acc_stage: int


DEFAULT_TUNE_SEARCH_SPACE: List[GemmTuneConfig] = []
for _mma_tiler_mn, _cluster_shape_mnk, _use_2cta_instrs, _num_ab_stage in product(
    [(128, 128), (128, 256), (256, 128), (256, 256)],
    [(1, 1, 1), (2, 1, 1)],
    [False, True],
    [4, 5],
):
    if _mma_tiler_mn[0] == 128 and _use_2cta_instrs:
        continue
    DEFAULT_TUNE_SEARCH_SPACE.append(
        GemmTuneConfig(
            mma_tiler_mn=_mma_tiler_mn,
            cluster_shape_mnk=_cluster_shape_mnk,
            use_2cta_instrs=_use_2cta_instrs,
            num_ab_stage=_num_ab_stage,
            num_acc_stage=1,
        )
    )

_CONFIG_KERNEL_CACHE: Dict[Tuple[Any, ...], Any] = {}
_INPUT_BEST_CONFIG_CACHE: Dict[Tuple[Any, ...], GemmTuneConfig] = {}
_PERSISTENT_CACHE_VERSION = "v1"
_PERSISTENT_CACHE_ENABLED = os.getenv("NVFP4_CUTE_PERSISTENT_CACHE", "0") == "1"
_PERSISTENT_CACHE_FUNC_NAME = "func"


def _get_persistent_cache_dir() -> Path:
    cache_dir = os.getenv("NVFP4_CUTE_CACHE_DIR")
    if cache_dir:
        path = Path(cache_dir)
    else:
        path = Path(tempfile.gettempdir()) / getuser() / "minitransformer_cute_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _persistent_cache_hash(config_key: Tuple[Any, ...]) -> str:
    payload = (_PERSISTENT_CACHE_VERSION, config_key)
    return hashlib.sha256(pickle.dumps(payload)).hexdigest()


class _FileLock:
    def __init__(self, lock_path: Path, exclusive: bool):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self._fd = None

    def __enter__(self):
        self._fd = os.open(
            str(self.lock_path),
            os.O_CREAT | (os.O_WRONLY if self.exclusive else os.O_RDONLY),
        )
        mode = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        fcntl.flock(self._fd, mode)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


def _try_load_compiled_from_disk(config_key: Tuple[Any, ...]):
    if not _PERSISTENT_CACHE_ENABLED:
        return None
    cache_dir = _get_persistent_cache_dir()
    key_hash = _persistent_cache_hash(config_key)
    obj_path = cache_dir / f"{key_hash}.o"
    lock_path = cache_dir / f"{key_hash}.lock"
    with _FileLock(lock_path, exclusive=False):
        if not obj_path.exists():
            return None
        module = cute.runtime.load_module(str(obj_path), enable_tvm_ffi=True)
        return getattr(module, _PERSISTENT_CACHE_FUNC_NAME)


def _try_export_compiled_to_disk(config_key: Tuple[Any, ...], compiled) -> None:
    if not _PERSISTENT_CACHE_ENABLED:
        return
    cache_dir = _get_persistent_cache_dir()
    key_hash = _persistent_cache_hash(config_key)
    obj_path = cache_dir / f"{key_hash}.o"
    lock_path = cache_dir / f"{key_hash}.lock"
    with _FileLock(lock_path, exclusive=True):
        if obj_path.exists():
            return
        compiled.export_to_c(
            object_file_path=str(obj_path),
            function_name=_PERSISTENT_CACHE_FUNC_NAME,
        )

class OptimizedBlockScaledDenseGemmKernel:
    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int] = DEFAULT_MMA_TILER_MN,
        cluster_shape_mnk: Tuple[int, int, int] = DEFAULT_CLUSTER_SHAPE_MNK,
        use_2cta_instrs: Optional[bool] = None,
        num_ab_stage: int = DEFAULT_NUM_AB_STAGE,
        num_acc_stage: int = DEFAULT_NUM_ACC_STAGE,
    ):
        self.threads_per_cta = 128
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = 512

        # set stages for ab_pipeline and acc_pipeline
        self.num_acc_stage = num_acc_stage
        self.num_ab_stage = num_ab_stage
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mnk = cluster_shape_mnk
        self.mma_inst_shape_k = mma_inst_shape_k
        self.use_2cta_instrs = (
            (mma_tiler_mn[0] != 128) if use_2cta_instrs is None else use_2cta_instrs
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        problem_size: tuple,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        accumulate: cutlass.Constexpr = False,
    ):
        # setup static attributes before smem/grid/tma computation
        self.c_layout = utils.LayoutEnum.ROW_MAJOR
        m, n, k, l = problem_size

        # Setup attributes that depend on gemm inputs
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler_mn[0],
            self.mma_tiler_mn[1],
            self.mma_inst_shape_k * mma_inst_tile_k,
        )

        self.mma_inst_shape_sfb = (
            self.mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1),
            self.mma_tiler_mn[1],
            self.mma_inst_shape_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_inst_shape_k * mma_inst_tile_k,
        )

        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n * k, 32)),
            ),
        )
        # 256bit aligned. row_major
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout(
                (cute.assume(m, 32), cute.assume(n, 16), l),
                stride=(cute.assume(n, 16), 1, cute.assume(m * n, 512)),
            ),
        )

        # Setup sfa/sfb tensor by filling A/B tensor to scale factor atom layout
        # ((Atom_M, Rest_M),(Atom_K, Rest_K),RestL)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        mma_op = tcgen05.MmaMXF4NVF4Op(
            sf_dtype,
            (*self.mma_tiler_mn, self.mma_inst_shape_k),
            tcgen05.CtaGroup.ONE if not self.use_2cta_instrs else tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)

        # (CTA_Tile_Shape_M, Round_Up(MMA_Tile_Shape_N, 128), MMA_Inst_Shape_K)
        # Note sfB don't support share among 2ctas
        sfb_mma_op = tcgen05.MmaMXF4NVF4Op(
            sf_dtype,
            self.mma_inst_shape_sfb,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
        )
        tiled_mma_sfb = cute.make_tiled_mma(sfb_mma_op)

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // (2 if self.use_2cta_instrs else 1),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // (2 if self.use_2cta_instrs else 1),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Compute A/B/SFA/SFB/C shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            self.num_ab_stage,
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        # TMA load for A
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load for SFA
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # Compute TMA load bytes
        a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Compute grid size
        grid = cute.round_up(
            cute.ceil_div(
                (c_tensor.layout.shape),
                (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1], 1),
            ),
            self.cluster_shape_mnk,
        )

        # Launch the kernel
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c_tensor,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            epilogue_op,
            accumulate,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
        )
        return

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        cta_layout_vmnk: cute.Layout,
        cta_layout_sfb_vmnk: cute.Layout,
        epilogue_op: cutlass.Constexpr,
        accumulate: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        cta_in_cluster_coord_sfb_vmnk = cta_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coords outside cluster
        mma_tile_coord_vmnk = (
            bidx % cute.size(cta_layout_vmnk, mode=[0]),
            bidx // cute.size(cta_layout_vmnk, mode=[0]),
            bidy,
            bidz,
        )
        mma_tile_coord_mnl = mma_tile_coord_vmnk[1:]
        is_leader_cta = mma_tile_coord_vmnk[0] == 0

        #
        # Define shared storage for kernel
        #
        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sSFA = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sSFB = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        #
        # Compute multicast mask for A/B/SFA/SFB buffer full
        #
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(
            self.is_a_mcast or self.is_b_mcast or self.use_2cta_instrs
        ):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_vmnk, cta_in_cluster_coord_vmnk, mcast_mode=2
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cta_layout_sfb_vmnk, cta_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Initialize mainloop ab_pipeline, acc_pipeline and their states
        #
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads_per_cta * (2 if self.use_2cta_instrs else 1),
            ),
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, RestM, RestK, RestL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        # (bN, bK, RestN, RestK, RestL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        # (bM, bN, RestM, RestN, RestL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        #
        # Partition global tensor for TiledMMA_A/B/SFA/SFB/C
        #
        # (MMA, MMA_M, MMA_K, RestK)
        thr_mma = tiled_mma.get_slice(mma_tile_coord_vmnk[0])
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_vmnk[0])
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgA = thr_mma.partition_A(gA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_M, MMA_K, RestM, RestK, RestL)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        # tCgSFB = thr_mma.partition_B(gSFB_nkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, RestM, RestN, RestL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load A/B/SFA/SFB
        #
        # TMA load A partition_S/D
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            # 0,
            # cute.make_layout(1),
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        # TMA load B partition_S/D
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            # 0,
            # cute.make_layout(1),
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        #  TMA load SFA partition_S/D
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestM, RestK, RestL)
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            # 0,
            # cute.make_layout(1),
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # TMA load SFB partition_S/D
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            cta_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N)
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        #
        # Alloc tensor memory buffer
        #
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        #
        # Make SFA/SFB tmem tensor
        #
        # Get SFA tmem ptr
        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc),
            dtype=sf_dtype,
        )
        # (MMA, MMA_M, MMA_K)
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        # Get SFB tmem ptr
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=sf_dtype,
        )
        # (MMA, MMA_N, MMA_K)
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
        )
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        #
        # Partition for S2T copy of SFA/SFB
        #
        # Make S2T CopyAtom
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(
                tcgen05.CtaGroup.ONE
                if not self.use_2cta_instrs
                else tcgen05.CtaGroup.TWO
            ),
            sf_dtype,
        )
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSFA_compact = cute.filter_zeros(sSFA)
        # (MMA, MMA_MN, MMA_K)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSFB_compact = cute.filter_zeros(sSFB)
        # (MMA, MMA_MN, MMA_K)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

        #
        # Slice to per mma tile index
        #
        # ((atom_v, rest_v), RestK)
        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        # ((atom_v, rest_v), RestK)
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        # ((atom_v, rest_v), RestK)
        tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        # ((atom_v, rest_v), RestK)
        tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        #
        # Execute Data copy and Math computation in the k_tile loop
        #
        if warp_idx == 0:
            # Wait for accumulator buffer empty
            if is_leader_cta:
                acc_producer.acquire_and_advance()

            # Set ACCUMULATE field to False for the first k_tile iteration
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            # Execute k_tile loop
            for k_tile in cutlass.range(
                k_tile_cnt, prefetch_stages=self.num_ab_stage - 2
            ):
                # Wait for AB buffer empty
                ab_empty = ab_producer.acquire_and_advance()

                #  TMA load A/B/SFA/SFB
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, ab_empty.count)],
                    tAsA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, ab_empty.count)],
                    tBsB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=b_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfa,
                    tAgSFA[(None, ab_empty.count)],
                    tAsSFA[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=sfa_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB[(None, ab_empty.count)],
                    tBsSFB[(None, ab_empty.index)],
                    tma_bar_ptr=ab_empty.barrier,
                    mcast_mask=sfb_full_mcast_mask,
                )

                if is_leader_cta:
                    # Wait for AB buffer full
                    ab_full = ab_consumer.wait_and_advance()

                    #  Copy SFA/SFB to tmem
                    s2t_stage_coord = (None, None, None, None, ab_full.index)
                    tCsSFA_compact_s2t_staged = tCsSFA_compact_s2t[s2t_stage_coord]
                    tCsSFB_compact_s2t_staged = tCsSFB_compact_s2t[s2t_stage_coord]
                    cute.copy(
                        tiled_copy_s2t_sfa,
                        tCsSFA_compact_s2t_staged,
                        tCtSFA_compact_s2t,
                    )
                    cute.copy(
                        tiled_copy_s2t_sfb,
                        tCsSFB_compact_s2t_staged,
                        tCtSFB_compact_s2t,
                    )

                    # tCtAcc += tCrA * tCrSFA * tCrB * tCrSFB
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                        kblock_coord = (
                            None,
                            None,
                            kblock_idx,
                            ab_full.index,
                        )

                        # Set SFA/SFB tensor to tiled_mma
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(
                            tcgen05.Field.SFA,
                            tCtSFA[sf_kblock_coord].iterator,
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB,
                            tCtSFB[sf_kblock_coord].iterator,
                        )

                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[kblock_coord],
                            tCrB[kblock_coord],
                            tCtAcc,
                        )
                        # Enable accumulate on tCtAcc after first kblock
                        if kblock_idx == 0:
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    # Async arrive AB buffer empty
                    ab_full.release()
            if is_leader_cta:
                acc_producer.commit()

        #
        # Epilogue
        # Partition for epilogue
        #
        # x32 or x128 all is ok.
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, cutlass.Float32)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R_M, T2R_N, EPI_M, EPI_M)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
        # (T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
        tTR_gC = thr_copy_t2r.partition_D(tCgC)
        c_elem_dtype = mC_mnl.element_type
        # (T2R_M, T2R_N, EPI_M, EPI_N）
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
        )
        # (T2R_M, T2R_N, EPI_M, EPI_N）
        tTR_rC = cute.make_rmem_tensor(
            tTR_gC[None, None, None, None, 0, 0, 0].shape, c_elem_dtype
        )
        # STG Atom
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_elem_dtype)
        tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]

        # Wait for accumulator buffer full
        acc_full = acc_consumer.wait_and_advance()

        # Copy accumulator to register
        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        acc_vec = epilogue_op(tTR_rAcc.load())
        if cutlass.const_expr(accumulate):
            # Inplace accumulate: C <- C + acc
            cute.copy(simt_atom, tTR_gC, tTR_rC)
            c_global = tTR_rC.load()
            acc_vec = (acc_vec + c_global)
        tTR_rC.store(acc_vec.to(c_elem_dtype))
        # Store C to global memory
        cute.copy(simt_atom, tTR_rC, tTR_gC)

        acc_full.release()

        # Ensure used buffers are properly synchronized before producer exit.
        # This could avoid the invalid dsmem access due to early leading CTA exit.
        if warp_idx == 0:
            ab_producer.tail()
            if is_leader_cta:
                acc_producer.tail()

        # Deallocate TMEM
        cute.arch.barrier()
        tmem.free(acc_tmem_ptr)

        return


def _get_device_arch_from_torch_device(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


def _compile_kernel_with_config(
    config: GemmTuneConfig,
    *,
    a_ptr,
    b_ptr,
    sfa_ptr,
    sfb_ptr,
    c_ptr,
    problem_size,
    stream,
    epilogue_op,
    epilogue_cache_tag: Any,
    out_dtype,
    accumulate: bool,
    has_swizzle: bool,
    arch: str,
):
    config_key = (
        arch,
        ab_dtype,
        sf_dtype,
        out_dtype,
        bool(accumulate),
        bool(has_swizzle),
        epilogue_cache_tag,
        config.mma_tiler_mn,
        config.cluster_shape_mnk,
        config.use_2cta_instrs,
        config.num_ab_stage,
        config.num_acc_stage,
    )
    if config_key in _CONFIG_KERNEL_CACHE:
        return _CONFIG_KERNEL_CACHE[config_key]

    compiled = _try_load_compiled_from_disk(config_key)
    if compiled is not None:
        _CONFIG_KERNEL_CACHE[config_key] = compiled
        return compiled

    gemm = OptimizedBlockScaledDenseGemmKernel(
        mma_tiler_mn=config.mma_tiler_mn,
        cluster_shape_mnk=config.cluster_shape_mnk,
        use_2cta_instrs=config.use_2cta_instrs,
        num_ab_stage=config.num_ab_stage,
        num_acc_stage=config.num_acc_stage,
    )
    compiled = cute.compile(
        gemm,
        a_ptr,
        b_ptr,
        sfa_ptr,
        sfb_ptr,
        c_ptr,
        problem_size,
        stream,
        epilogue_op=epilogue_op,
        accumulate=accumulate,
    )
    _try_export_compiled_to_disk(config_key, compiled)
    _CONFIG_KERNEL_CACHE[config_key] = compiled
    return compiled


def get_compiled_nvfp4_gemm(
    *,
    a_ptr,
    b_ptr,
    sfa_ptr,
    sfb_ptr,
    c_ptr,
    problem_size: Tuple[int, int, int, int],
    stream,
    epilogue_op,
    epilogue_cache_tag: Any = None,
    out_dtype=torch.float16,
    accumulate: bool = False,
    has_swizzle: bool = True,
    do_autotune: bool = False,
    warmup_iterations: int = 10,
    iterations: int = 100,
    use_cold_l2: bool = True,
    benchmark_scope_tensors: Optional[List[torch.Tensor]] = None,
    tune_search_space: Optional[List[GemmTuneConfig]] = None,
):
    if tune_search_space is None:
        tune_search_space = DEFAULT_TUNE_SEARCH_SPACE
    if len(tune_search_space) == 0:
        raise ValueError("tune_search_space must not be empty")

    if not isinstance(c_ptr, cute.Pointer):
        raise TypeError("c_ptr must be a cute.Pointer")
    torch_device = torch.device("cuda", torch.cuda.current_device())
    arch = _get_device_arch_from_torch_device(torch_device)
    m, n, k, l = problem_size

    input_key = (
        torch_device.index,
        arch,
        m,
        n,
        k,
        out_dtype,
        bool(accumulate),
        bool(has_swizzle),
        epilogue_cache_tag,
    )

    def _benchmark(compiled, scope_tensors):
        def workspace_generator():
            args = cute.testing.JitArguments(
                a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, problem_size, stream
            )
            args.add_to_scope(scope_tensors)
            return args

        workspace_count = 1
        if use_cold_l2:
            one_workspace_bytes = sum(
                int(t.numel() * t.element_size()) for t in scope_tensors
            )
            workspace_count = cute.testing.get_workspace_count(
                one_workspace_bytes, warmup_iterations, iterations
            )
        return cute.testing.benchmark(
            compiled,
            workspace_generator=workspace_generator,
            workspace_count=workspace_count,
            stream=stream,
            warmup_iterations=warmup_iterations,
            iterations=iterations,
        )

    if do_autotune:
        if benchmark_scope_tensors is None or len(benchmark_scope_tensors) == 0:
            raise ValueError("benchmark_scope_tensors is required when do_autotune=True")
        if input_key in _INPUT_BEST_CONFIG_CACHE:
            best_config = _INPUT_BEST_CONFIG_CACHE[input_key]
            compiled = _compile_kernel_with_config(
                best_config,
                a_ptr=a_ptr,
                b_ptr=b_ptr,
                sfa_ptr=sfa_ptr,
                sfb_ptr=sfb_ptr,
                c_ptr=c_ptr,
                problem_size=problem_size,
                stream=stream,
                epilogue_op=epilogue_op,
                epilogue_cache_tag=epilogue_cache_tag,
                out_dtype=out_dtype,
                accumulate=accumulate,
                has_swizzle=has_swizzle,
                arch=arch,
            )
            return compiled, best_config, None

        best_time = float("inf")
        best_config = None
        best_compiled = None

        for config in tune_search_space:
            if config.use_2cta_instrs and config.mma_tiler_mn[0] == 128:
                continue
            try:
                compiled = _compile_kernel_with_config(
                    config,
                    a_ptr=a_ptr,
                    b_ptr=b_ptr,
                    sfa_ptr=sfa_ptr,
                    sfb_ptr=sfb_ptr,
                    c_ptr=c_ptr,
                    problem_size=problem_size,
                    stream=stream,
                    epilogue_op=epilogue_op,
                    epilogue_cache_tag=epilogue_cache_tag,
                    out_dtype=out_dtype,
                    accumulate=accumulate,
                    has_swizzle=has_swizzle,
                    arch=arch,
                )
                elapsed_us = _benchmark(compiled, benchmark_scope_tensors)
            except Exception:
                continue
            if elapsed_us < best_time:
                best_time = elapsed_us
                best_config = config
                best_compiled = compiled

        if best_compiled is None:
            raise RuntimeError("autotune failed: no valid NVFP4 CuTeDSL kernel config")

        _INPUT_BEST_CONFIG_CACHE[input_key] = best_config
        return best_compiled, best_config, best_time

    default_config = GemmTuneConfig(
        mma_tiler_mn=DEFAULT_MMA_TILER_MN,
        cluster_shape_mnk=DEFAULT_CLUSTER_SHAPE_MNK,
        use_2cta_instrs=(DEFAULT_MMA_TILER_MN[0] != 128),
        num_ab_stage=DEFAULT_NUM_AB_STAGE,
        num_acc_stage=DEFAULT_NUM_ACC_STAGE,
    )
    compiled = _compile_kernel_with_config(
        default_config,
        a_ptr=a_ptr,
        b_ptr=b_ptr,
        sfa_ptr=sfa_ptr,
        sfb_ptr=sfb_ptr,
        c_ptr=c_ptr,
        problem_size=problem_size,
        stream=stream,
        epilogue_op=epilogue_op,
        epilogue_cache_tag=epilogue_cache_tag,
        out_dtype=out_dtype,
        accumulate=accumulate,
        has_swizzle=has_swizzle,
        arch=arch,
    )
    return compiled, default_config, None
