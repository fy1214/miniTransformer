# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# NVFP4 variant of fp16_gemm_3.py: static persistent tile scheduling (no CLC) plus
# warp-specialized TMA / MMA / epilogue. C is written with SIMT stores like
# nvfp4_gemm_2.py (no TMA S2G for batched NVFP4 C layouts).

import argparse
import os
import sys
from typing import Tuple

import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(current_dir, "..", "..")
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

mma_tiler_mn = (256, 256)
mma_inst_shape_k = 64
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16
cluster_shape_mnk = (2, 1, 1)

ab_stages = 5
acc_stages = 1

scheduler_type = utils.StaticPersistentTileScheduler

"""
NVFP4 block-scaled GEMM with static persistent scheduling (fp16_gemm_3 structure)
and NVFP4 warp-specialized mainloop (nvfp4_gemm_2). Epilogue is SIMT.

Constraints match nvfp4_gemm_2. The scheduler grid includes the batch dimension
via ceil_div(l, 1), same as nvfp4_gemm_3_1 / nvfp4_gemm_2.
"""


class Sm100BlockScaledDenseGemm3Kernel:
    def __init__(self):
        self.num_tmem_alloc_cols = 512

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
        max_active_clusters: cutlass.Constexpr,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        accumulate: cutlass.Constexpr = False,
    ):
        m, n, k, l = problem_size

        self.use_2cta_instrs = False if mma_tiler_mn[0] == 128 else True

        mma_inst_tile_k = 4
        self.mma_tiler = (
            mma_tiler_mn[0],
            mma_tiler_mn[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.mma_inst_shape_sfb = (
            mma_tiler_mn[0] // (2 if self.use_2cta_instrs else 1),
            mma_tiler_mn[1],
            mma_inst_shape_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            mma_inst_shape_k * mma_inst_tile_k,
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
        c_tensor = cute.make_tensor(
            c_ptr,
            cute.make_layout(
                (cute.assume(m, 32), cute.assume(n, 16), l),
                stride=(cute.assume(n, 16), 1, cute.assume(m * n, 512)),
            ),
        )

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        mma_op = tcgen05.MmaMXF4NVF4Op(
            sf_dtype,
            (*mma_tiler_mn, mma_inst_shape_k),
            tcgen05.CtaGroup.ONE if not self.use_2cta_instrs else tcgen05.CtaGroup.TWO,
            tcgen05.OperandSource.SMEM,
        )
        tiled_mma = cute.make_tiled_mma(mma_op)

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

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(cluster_shape_mnk),
            (tiled_mma.thr_id.shape,),
        )
        cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout(cluster_shape_mnk),
            (tiled_mma_sfb.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            ab_stages,
        )
        b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            ab_dtype,
            ab_stages,
        )
        sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            ab_stages,
        )
        sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            sf_vec_size,
            ab_stages,
        )

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a_tensor,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b_tensor,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa_tensor,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            cluster_shape_mnk[:2], tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb_tensor,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        a_copy_size = cute.size_in_bytes(ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(ab_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

        # Batched C is 3D; avoid zipped_divide on c_tensor. Match nvfp4_gemm_2 grid:
        # ceil_div(shape, (cta_m, cta_n, 1)) => L dimension has one CTA tile per batch index.
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (
                cute.ceil_div(m, self.cta_tile_shape_mnk[0]),
                cute.ceil_div(n, self.cta_tile_shape_mnk[1]),
                cute.ceil_div(l, 1),
            ),
            cluster_shape_mnk,
        )
        grid_shape = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

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
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            cluster_layout_vmnk,
            cluster_layout_sfb_vmnk,
            tile_sched_params,
            epilogue_op,
            accumulate,
        ).launch(
            grid=grid_shape,
            block=[192, 1, 1],
            cluster=cluster_shape_mnk,
            stream=stream,
        )
        return

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
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        accumulate: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        epilogue_warp_ids = (0, 1, 2, 3)
        mma_warp_id = 4
        tma_warp_id = 5

        tmem_alloc_sync_bar_id = 2

        cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
        cta_in_cluster_coord_vmnk = cta_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        cta_in_cluster_coord_sfb_vmnk = cta_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        mma_tile_coord_v = bidx % cute.size(cta_layout_vmnk, mode=[0])
        is_leader_cta = mma_tile_coord_v == 0

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, acc_stages * 2]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        if warp_idx == tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        num_mcast_participants = (
            cute.size(cta_layout_vmnk, mode=[1])
            + cute.size(cta_layout_vmnk, mode=[2])
            - 1
        )

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

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=tmem_alloc_sync_bar_id,
            num_threads=32 * len((mma_warp_id, *epilogue_warp_ids)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=epilogue_warp_ids[0],
            is_two_cta=cute.size(cta_layout_vmnk, mode=[0]) > 1,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        num_tma_copy_bytes = self.num_tma_load_bytes
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            size=num_mcast_participants,
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=ab_stages,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=num_tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            (32 * len(epilogue_warp_ids))
            * cute.size(cta_layout_vmnk, mode=[0]),
        )
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=acc_stages,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
        ).make_participants()

        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mnk, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        num_k_tiles = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            cta_in_cluster_coord_vmnk[1],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[1])),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            cta_in_cluster_coord_vmnk[2],
            cute.make_layout(cute.size(cta_layout_vmnk, mode=[2])),
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

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

        pipeline_init_wait(cluster_shape_mn=cluster_shape_mnk)

        tile_sched = scheduler_type.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        if warp_idx == tma_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                tAgSFA_slice = tAgSFA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
                ]
                tBgSFB_slice = tBgSFB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]
                for k_tile_idx in range(num_k_tiles):
                    handle = ab_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, k_tile_idx)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, k_tile_idx)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_slice[(None, k_tile_idx)],
                        tAsSFA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=sfa_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_slice[(None, k_tile_idx)],
                        tBsSFB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=sfb_full_mcast_mask,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_producer.tail()

        if warp_idx == mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_head = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc_head),
                dtype=sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr
                + tcgen05.find_tmem_tensor_col_offset(tCtAcc_head)
                + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
                dtype=sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            copy_atom_s2t = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(
                    tcgen05.CtaGroup.ONE
                    if not self.use_2cta_instrs
                    else tcgen05.CtaGroup.TWO
                ),
                sf_dtype,
            )
            tCsSFA_compact = cute.filter_zeros(sSFA)
            tCtSFA_compact = cute.filter_zeros(tCtSFA)
            tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(
                copy_atom_s2t, tCtSFA_compact
            )
            thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
            tCsSFA_compact_s2t_ = thr_copy_s2t_sfa.partition_S(tCsSFA_compact)
            tCsSFA_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_sfa, tCsSFA_compact_s2t_
            )
            tCtSFA_compact_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

            tCsSFB_compact = cute.filter_zeros(sSFB)
            tCtSFB_compact = cute.filter_zeros(tCtSFB)
            tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(
                copy_atom_s2t, tCtSFB_compact
            )
            thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
            tCsSFB_compact_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
            tCsSFB_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
                tiled_copy_s2t_sfb, tCsSFB_compact_s2t_
            )
            tCtSFB_compact_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

            while work_tile.is_valid_tile:
                if is_leader_cta:
                    tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    acc_empty = acc_producer.acquire_and_advance()
                    for k_tile_idx in range(num_k_tiles):
                        handle = ab_consumer.wait_and_advance()
                        s2t_stage_coord = (None, None, None, None, handle.index)
                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[s2t_stage_coord],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[s2t_stage_coord],
                            tCtSFB_compact_s2t,
                        )
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, handle.index)
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
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        handle.release()
                    acc_empty.commit()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_producer.tail()

        if warp_idx < mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
            tCtAcc_epi = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            op_t2r = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
            copy_atom_t2r = cute.make_copy_atom(op_t2r, cutlass.Float32)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc_epi)
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc_epi)
            tTR_gC_base = thr_copy_t2r.partition_D(tCgC)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_gC_base[None, None, None, None, 0, 0, 0].shape, cutlass.Float32
            )
            c_elem_dtype = mC_mnl.element_type
            tTR_rC = cute.make_rmem_tensor(
                tTR_gC_base[None, None, None, None, 0, 0, 0].shape, c_elem_dtype
            )
            simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), c_elem_dtype)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                acc_full = acc_consumer.wait_and_advance()
                cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
                tTR_gC = tTR_gC_base[
                    (None, None, None, None, *mma_tile_coord_mnl)
                ]
                acc_vec = epilogue_op(tTR_rAcc.load().to(c_elem_dtype))
                if cutlass.const_expr(accumulate):
                    cute.copy(simt_atom, tTR_gC, tTR_rC)
                    c_global = tTR_rC.load()
                    acc_vec = acc_vec + c_global
                tTR_rC.store(acc_vec)
                cute.copy(simt_atom, tTR_rC, tTR_gC)
                acc_full.release()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        return

