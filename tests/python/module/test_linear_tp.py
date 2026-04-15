"""
Linear Tensor Parallel (TP) Accuracy Tests
===========================================

This module tests the accuracy of Linear layer with Tensor Parallel (TP) by
comparing outputs with tp_size=1 (baseline) and tp_size>1.

The tests verify:
1. Forward pass accuracy between tp=1 and tp>1
2. Backward pass gradient accuracy
3. Column parallel and Row parallel modes
4. Different quantization recipes
"""
import argparse
import os
import traceback
import torch
import torch.distributed as dist
from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.linear import Linear


# ==================== TP Utilities ====================

def initialize_distributed(rank, world_size):
    """Initialize distributed environment for TP testing."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
    
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def create_tp_group(tp_size):
    """Create a TP group for testing."""
    if not dist.is_initialized():
        return None
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if tp_size > world_size:
        raise ValueError(f"tp_size ({tp_size}) cannot be larger than world_size ({world_size})")
    
    # Create TP group
    ranks = list(range(tp_size))
    tp_group = dist.new_group(ranks=ranks)
    
    return tp_group if rank < tp_size else None


# ==================== Weight / Input Distribution Utilities ====================

def broadcast_tensor(tensor: torch.Tensor, tp_group) -> torch.Tensor:
    """Broadcast tensor from rank 0 to all ranks in tp_group."""
    dist.broadcast(tensor, src=0, group=tp_group)
    return tensor


def sync_linear_weight(linear, tp_group) -> None:
    """
    Broadcast linear.weight from rank 0 to all ranks in tp_group.

    Every rank builds its own linear layer (random init differs per rank).
    Calling this right after build ensures all ranks share the same baseline
    weight before any scatter / shard operations.
    """
    dist.broadcast(linear.weight.data, src=0, group=tp_group)


# ==================== Test Class ====================

class TestLinearTP:
    """Test Linear layer with Tensor Parallel."""
    
    def build_linear(
        self,
        in_features: int,
        out_features: int,
        parallel_mode: str,
        tp_size: int = 1,
        tp_group = None,
        qparam: QuantizeRecipe = None,
        seed: int = 42,
        sequence_parallel: bool = False,
    ):
        """
        Build a Linear layer with TP configuration.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            parallel_mode: 'column' or 'row'
            tp_size: Tensor parallel size
            tp_group: Tensor parallel process group
            qparam: Quantization recipe
            seed: Random seed for initialization
            sequence_parallel: Enable sequence parallel (SP). When True,
                column parallel AllGathers input along dim 0 before GEMM,
                and row parallel ReduceScatters output along dim 0 after GEMM.
            
        Returns:
            Linear layer
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        fuse_wgrad_accumulation = True
        get_rng_state_tracker = None
        rng_tracker_name = None
        init_method = None
        bias = False
        return_bias = False
        params_dtype = torch.bfloat16
        parameters_split = None
        device = "cuda"
        ub_overlap_rs = False
        ub_overlap_ag = False
        ub_name = f"linear_tp{tp_size}_{parallel_mode}"
        layer_number = 0
        qlinear_params = get_qlinear_params_from_predefined(qparam)
        
        linear = Linear(
            in_features,
            out_features,
            sequence_parallel,
            fuse_wgrad_accumulation,
            tp_group,
            tp_size,
            get_rng_state_tracker,
            rng_tracker_name,
            init_method,
            bias,
            return_bias,
            params_dtype,
            parallel_mode,
            parameters_split,
            device,
            ub_overlap_rs,
            ub_overlap_ag,
            ub_name,
            layer_number,
            qlinear_params=qlinear_params
        )
        
        return linear
    
    def test_column_parallel_forward(
        self, qparam: QuantizeRecipe, tp_size: int = 2, sequence_parallel: bool = True
    ):
        """
        Test column parallel forward pass accuracy.

        Column parallel splits out_features across TP ranks.

        SP=True:
          - Each rank receives inp[rank*chunk:(rank+1)*chunk, :] (sequence shard, dim 0)
          - linear_tp AllGathers full input internally before GEMM
          - Each rank outputs [batch, out_features/tp_size]
          - AllGather outputs across ranks -> [batch, out_features]

        SP=False:
          - Every rank receives the full input [batch, in_features]
          - No AllGather; each rank computes its own output shard directly
          - Each rank outputs [batch, out_features/tp_size]
          - AllGather outputs across ranks -> [batch, out_features]
        """
        print(f"\n{'='*60}")
        print(f"Testing Column Parallel Forward (tp_size={tp_size}, sp={sequence_parallel})")
        print(f"{'='*60}")

        in_features = 4096
        out_features = 4096
        batch_size = 1024
        seed = 42

        tp_group = create_tp_group(tp_size) if tp_size > 1 else None
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Build baseline (tp=1, sp=False)
        linear_baseline = self.build_linear(
            in_features, out_features,
            parallel_mode='column',
            tp_size=1,
            tp_group=None,
            qparam=qparam,
            seed=seed,
            sequence_parallel=False,
        )
        # Ensure all ranks have identical baseline weight (rank 0 is authoritative)
        if tp_size > 1:
            sync_linear_weight(linear_baseline, tp_group)

        # Build TP version
        linear_tp = self.build_linear(
            in_features, out_features,
            parallel_mode='column',
            tp_size=tp_size,
            tp_group=tp_group,
            qparam=qparam,
            seed=seed,
            sequence_parallel=sequence_parallel,
        )

        # linear_baseline.weight is already synced across all ranks.
        # Each rank copies its own shard directly (no communication needed).
        if tp_size > 1:
            chunk_out = out_features // tp_size
            with torch.no_grad():
                linear_tp.weight.data.copy_(
                    linear_baseline.weight.data[rank * chunk_out:(rank + 1) * chunk_out, :]
                )

        # Create full input on rank 0, broadcast so all ranks have identical data
        torch.manual_seed(seed)
        inp_full = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
        if tp_size > 1:
            broadcast_tensor(inp_full, tp_group)

        # SP=True : each rank gets its sequence shard (dim 0)
        # SP=False: every rank uses the full input
        if sequence_parallel and tp_size > 1:
            chunk_seq = batch_size // tp_size
            inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
        else:
            inp_tp = inp_full

        # Forward
        with torch.no_grad():
            out_baseline = linear_baseline(inp_full, is_first_microbatch=True)
            out_tp = linear_tp(inp_tp, is_first_microbatch=True)

        if rank < tp_size:
            # Column parallel: rank r holds output columns [r*chunk:(r+1)*chunk].
            # Compare out_tp (local shard) directly against the matching slice of
            # out_baseline. No AllGather needed — the column order already matches.
            tp_out = out_features // tp_size
            baseline_local = out_baseline[:, rank * tp_out:(rank + 1) * tp_out]

            abs_diff = (baseline_local - out_tp).abs()
            rel_diff = abs_diff / (baseline_local.abs() + 1e-8)

            print(f"Rank {rank}:")
            print(f"  Baseline output shape:        {out_baseline.shape}")
            print(f"  TP output shape (local):      {out_tp.shape}")
            print(f"  Max absolute diff:  {abs_diff.max().item():.6e}")
            print(f"  Mean absolute diff: {abs_diff.mean().item():.6e}")
            print(f"  Max relative diff:  {rel_diff.max().item():.6e}")

            torch.testing.assert_close(
                baseline_local, out_tp,
                rtol=1e-2, atol=1e-2,
                msg=f"Column parallel forward mismatch at rank {rank}"
            )
            print(f"✓ Column parallel forward pass accuracy verified!")

        print(f"{'='*60}\n")

    def test_row_parallel_forward(
        self, qparam: QuantizeRecipe, tp_size: int = 2, sequence_parallel: bool = True
    ):
        """
        Test row parallel forward pass accuracy.

        Row parallel 数学等价性：
          完整 GEMM: output = inp_full @ W_full.T   [m, n]
          分片 GEMM: output = sum_r(inp_r @ W_r.T)  [m, n]
            其中 inp_r = inp_full[:, r*k//tp:(r+1)*k//tp]  [m, k//tp]
                 W_r   = W_full[:, r*k//tp:(r+1)*k//tp]    [n, k//tp]

        非 SP：AllReduce 后 out_tp = sum，直接与 out_baseline 比较
        SP   ：ReduceScatter 后 out_tp = [m//tp, n]（sequence shard），
               与 out_baseline[rank*chunk:(rank+1)*chunk, :] 比较
        """
        print(f"\n{'='*60}")
        print(f"Testing Row Parallel Forward (tp_size={tp_size}, sp={sequence_parallel})")
        print(f"{'='*60}")

        in_features = 4096
        out_features = 4096
        batch_size = 1024
        seed = 42

        tp_group = create_tp_group(tp_size) if tp_size > 1 else None
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Build baseline (tp=1, sp=False)
        linear_baseline = self.build_linear(
            in_features, out_features,
            parallel_mode='row',
            tp_size=1,
            tp_group=None,
            qparam=qparam,
            seed=seed,
            sequence_parallel=False,
        )
        # Ensure all ranks have identical baseline weight (rank 0 is authoritative)
        if tp_size > 1:
            sync_linear_weight(linear_baseline, tp_group)

        # Build TP version
        linear_tp = self.build_linear(
            in_features, out_features,
            parallel_mode='row',
            tp_size=tp_size,
            tp_group=tp_group,
            qparam=qparam,
            seed=seed,
            sequence_parallel=sequence_parallel,
        )

        # linear_baseline.weight shape: [out_features, in_features]
        # row parallel splits in_features (dim 1)，linear_tp.weight shape: [out_features, in_features//tp]
        if tp_size > 1:
            chunk_in = in_features // tp_size
            with torch.no_grad():
                linear_tp.weight.data.copy_(
                    linear_baseline.weight.data[:, rank * chunk_in:(rank + 1) * chunk_in]
                )

        # Create full input on rank 0, broadcast to all ranks
        # 使用整数值初始化 x, w ∈ {-1, 0, 1}，乘积和累加均为整数，
        # bf16 可精确表示，避免浮点累加误差导致 bit-exact 验证失败
        torch.manual_seed(seed)
        inp_full = torch.randint(-1, 2, (batch_size, in_features), device="cuda").to(torch.bfloat16)
        if tp_size > 1:
            broadcast_tensor(inp_full, tp_group)

        # 同样将 weight 初始化为整数值（需要在 sync/copy 之后覆盖）
        with torch.no_grad():
            linear_baseline.weight.data.copy_(
                torch.randint(-1, 2, linear_baseline.weight.shape, device="cuda").to(torch.bfloat16)
            )
        if tp_size > 1:
            sync_linear_weight(linear_baseline, tp_group)
            chunk_in = in_features // tp_size
            linear_tp.weight.data.copy_(
                linear_baseline.weight.data[:, rank * chunk_in:(rank + 1) * chunk_in]
            )

        # Row parallel: 每个 rank 输入是 inp_full 的 in_features 切片 [m, k//tp]
        # 注意：无论 SP 是否开启，row parallel 的输入切分方式都是按 in_features 维度
        if tp_size > 1:
            chunk_in = in_features // tp_size
            inp_tp = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous()
        else:
            inp_tp = inp_full

        # Forward
        with torch.no_grad():
            # baseline: inp_full [m, k] @ W_full.T [k, n] = [m, n]
            out_baseline = linear_baseline(inp_full, is_first_microbatch=True)
            # tp: inp_tp [m, k//tp] @ W_r.T [k//tp, n] -> AllReduce -> [m, n]
            #  or ReduceScatter -> [m//tp, n] if SP
            out_tp = linear_tp(inp_tp, is_first_microbatch=True)

        if rank < tp_size:
            # 非 SP：out_tp = AllReduce(sum of partial GEMMs) = [m, n]，与 out_baseline 直接比
            # SP   ：out_tp = ReduceScatter -> [m//tp, n]，与 out_baseline[rank_seq_shard] 比
            if sequence_parallel and tp_size > 1:
                chunk_seq = batch_size // tp_size
                baseline_local = out_baseline[rank * chunk_seq:(rank + 1) * chunk_seq, :]
            else:
                baseline_local = out_baseline

            abs_diff = (baseline_local - out_tp).abs()

            print(f"Rank {rank}:")
            print(f"  inp_full shape:               {inp_full.shape}")
            print(f"  inp_tp shape (this rank):     {inp_tp.shape}")
            print(f"  Baseline output shape:        {out_baseline.shape}")
            print(f"  TP output shape (local):      {out_tp.shape}")
            print(f"  Max absolute diff:  {abs_diff.max().item():.6e}")
            print(f"  Mean absolute diff: {abs_diff.mean().item():.6e}")

            torch.testing.assert_close(
                baseline_local, out_tp,
                rtol=0, atol=0,
                msg=f"Row parallel forward mismatch at rank {rank}"
            )
            print(f"✓ Row parallel forward pass accuracy verified!")

        print(f"{'='*60}\n")

    def test_column_parallel_backward(
        self, qparam: QuantizeRecipe, tp_size: int = 2, sequence_parallel: bool = True
    ):
        """
        Test column parallel backward pass gradient accuracy.

        Column parallel backward 数学等价性：

        dgrad（dx = dy * W_full.T）：
          tp=1: dx = dy_full * W_full.T              [m, k]
          tp=2: dx_r = dy_r * W_r.T, AllReduce -> dx [m, k]
          非 SP: dx_tp == dx_baseline
          SP  : dx_tp == dx_baseline[seq_shard, :]

        wgrad（dw_r = dy_r.T * x_full）：
          tp=1: dw = dy_full.T * x_full              [n, k]
          tp=2: dw_r = dy_r.T * x_full               [n//tp, k]
          比较: dw_tp == dw_baseline[col_shard, :]

        使用整数初始化（x, w ∈ {-1,0,1}，dy=ones），所有梯度均为整数，
        bf16 精确表示，bit-exact 验证。
        """
        print(f"\n{'='*60}")
        print(f"Testing Column Parallel Backward (tp_size={tp_size}, sp={sequence_parallel})")
        print(f"{'='*60}")

        in_features = 4096
        out_features = 4096
        batch_size = 1024
        seed = 42

        tp_group = create_tp_group(tp_size) if tp_size > 1 else None
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Build baseline (tp=1, sp=False)
        linear_baseline = self.build_linear(
            in_features, out_features,
            parallel_mode='column',
            tp_size=1,
            tp_group=None,
            qparam=qparam,
            seed=seed,
            sequence_parallel=False,
        )
        if tp_size > 1:
            sync_linear_weight(linear_baseline, tp_group)
        linear_baseline.weight.main_grad = torch.zeros_like(
            linear_baseline.weight.data, dtype=torch.float32
        )

        # Build TP version
        linear_tp = self.build_linear(
            in_features, out_features,
            parallel_mode='column',
            tp_size=tp_size,
            tp_group=tp_group,
            qparam=qparam,
            seed=seed,
            sequence_parallel=sequence_parallel,
        )
        linear_tp.weight.main_grad = torch.zeros_like(
            linear_tp.weight.data, dtype=torch.float32
        )

        # 整数初始化 weight（x,w ∈ {-1,0,1}），bit-exact 验证
        torch.manual_seed(seed)
        with torch.no_grad():
            linear_baseline.weight.data.copy_(
                torch.randint(-1, 2, linear_baseline.weight.shape, device="cuda").to(torch.bfloat16)
            )
        if tp_size > 1:
            sync_linear_weight(linear_baseline, tp_group)
            chunk_out = out_features // tp_size
            linear_tp.weight.data.copy_(
                linear_baseline.weight.data[rank * chunk_out:(rank + 1) * chunk_out, :]
            )
            linear_tp.weight.main_grad = torch.zeros_like(
                linear_tp.weight.data, dtype=torch.float32
            )

        # 整数初始化 input，broadcast 保证一致
        inp_full = torch.randint(-1, 2, (batch_size, in_features), device="cuda").to(torch.bfloat16)
        if tp_size > 1:
            broadcast_tensor(inp_full, tp_group)

        inp_baseline = inp_full.clone().requires_grad_(True)

        # SP=True : each rank gets its sequence shard
        # SP=False: every rank gets the full input
        if sequence_parallel and tp_size > 1:
            chunk_seq = batch_size // tp_size
            inp_tp = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous().requires_grad_(True)
        else:
            inp_tp = inp_full.clone().requires_grad_(True)

        # 显式初始化 dy（整数值），broadcast 保证两边一致
        # dy_full shape: [batch, out_features]，与 baseline output shape 一致
        dy_full = torch.randint(-1, 2, (batch_size, out_features), device="cuda").to(torch.bfloat16)
        if tp_size > 1:
            broadcast_tensor(dy_full, tp_group)

        # baseline dy: 完整 [batch, out_features]
        # tp dy: 对应 out_features 列切片（column parallel 每个 rank 只输出自己的列分片）
        #   非SP: out_tp shape [batch, out//tp]       -> dy_tp = dy_full[:, col_shard]
        #   SP  : out_tp shape [batch//tp, out//tp]   -> dy_tp = dy_full[seq_shard, col_shard]
        chunk_out = out_features // tp_size
        if sequence_parallel and tp_size > 1:
            chunk_seq = batch_size // tp_size
            dy_tp = dy_full[rank * chunk_seq:(rank + 1) * chunk_seq,
                            rank * chunk_out:(rank + 1) * chunk_out].contiguous()
        else:
            dy_tp = dy_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous()

        # Forward + backward
        out_baseline = linear_baseline(inp_baseline, is_first_microbatch=True)
        out_tp       = linear_tp(inp_tp, is_first_microbatch=True)

        out_baseline.backward(dy_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous())
        out_tp.backward(dy_tp)

        if rank < tp_size:
            chunk_out = out_features // tp_size

            # --- dgrad 验证 ---
            if inp_baseline.grad is not None and inp_tp.grad is not None:
                if sequence_parallel and tp_size > 1:
                    chunk_seq = batch_size // tp_size
                    baseline_dgrad = inp_baseline.grad[rank * chunk_seq:(rank + 1) * chunk_seq, :]
                else:
                    baseline_dgrad = inp_baseline.grad

                abs_diff = (baseline_dgrad - inp_tp.grad).abs()
                print(f"Rank {rank} - dgrad:")
                print(f"  baseline_dgrad shape: {baseline_dgrad.shape}")
                print(f"  inp_tp.grad shape:    {inp_tp.grad.shape}")
                print(f"  Max absolute diff:    {abs_diff.max().item():.6e}")
                print(f"  Mean absolute diff:   {abs_diff.mean().item():.6e}")

                torch.testing.assert_close(
                    baseline_dgrad, inp_tp.grad,
                    rtol=0, atol=0,
                    msg=f"Column parallel dgrad mismatch at rank {rank}"
                )

            # --- wgrad 验证 ---
            if linear_baseline.weight.main_grad is not None and linear_tp.weight.main_grad is not None:
                baseline_wgrad = linear_baseline.weight.main_grad[
                    rank * chunk_out:(rank + 1) * chunk_out, :
                ]
                abs_diff = (baseline_wgrad - linear_tp.weight.main_grad).abs()
                print(f"Rank {rank} - wgrad:")
                print(f"  baseline_wgrad shape: {baseline_wgrad.shape}")
                print(f"  tp wgrad shape:       {linear_tp.weight.main_grad.shape}")
                print(f"  Max absolute diff:    {abs_diff.max().item():.6e}")
                print(f"  Mean absolute diff:   {abs_diff.mean().item():.6e}")

                torch.testing.assert_close(
                    baseline_wgrad, linear_tp.weight.main_grad,
                    rtol=0, atol=0,
                    msg=f"Column parallel wgrad mismatch at rank {rank}"
                )

            print(f"✓ Column parallel backward pass accuracy verified!")

        print(f"{'='*60}\n")

    def test_row_parallel_backward(
        self, qparam: QuantizeRecipe, tp_size: int = 2, sequence_parallel: bool = True
    ):
        """
        Test row parallel backward pass gradient accuracy.

        Row parallel backward 数学等价性：

        dgrad（dx_r = dy * W_r.T）：
          tp=1: dx = dy_full * W_full.T              [m, k]
          tp=2: dx_r = dy_full * W_r.T               [m, k//tp]
          比较: dx_tp == dx_baseline[:, in_shard]
          （row parallel dgrad 不需要通信，每个 rank 直接算自己的 in_features 分片）

        wgrad（dw_r = dy.T * x_r）：
          tp=1: dw = dy_full.T * x_full              [n, k]
          tp=2: dw_r = dy_full.T * x_r               [n, k//tp]
          SP  : dy 先 AllGather 再算 wgrad
          比较: dw_tp == dw_baseline[:, in_shard]

        使用整数初始化（x, w ∈ {-1,0,1}，dy=ones），所有梯度均为整数，
        bf16 精确表示，bit-exact 验证。
        """
        print(f"\n{'='*60}")
        print(f"Testing Row Parallel Backward (tp_size={tp_size}, sp={sequence_parallel})")
        print(f"{'='*60}")

        in_features = 4096
        out_features = 4096
        batch_size = 1024
        seed = 42

        tp_group = create_tp_group(tp_size) if tp_size > 1 else None
        rank = dist.get_rank() if dist.is_initialized() else 0

        # Build baseline (tp=1, sp=False)
        linear_baseline = self.build_linear(
            in_features, out_features,
            parallel_mode='row',
            tp_size=1,
            tp_group=None,
            qparam=qparam,
            seed=seed,
            sequence_parallel=False,
        )
        if tp_size > 1:
            sync_linear_weight(linear_baseline, tp_group)
        linear_baseline.weight.main_grad = torch.zeros_like(
            linear_baseline.weight.data, dtype=torch.float32
        )

        # Build TP version
        linear_tp = self.build_linear(
            in_features, out_features,
            parallel_mode='row',
            tp_size=tp_size,
            tp_group=tp_group,
            qparam=qparam,
            seed=seed,
            sequence_parallel=sequence_parallel,
        )
        linear_tp.weight.main_grad = torch.zeros_like(
            linear_tp.weight.data, dtype=torch.float32
        )

        # 整数初始化 weight（{-1,0,1}），bit-exact 验证
        torch.manual_seed(seed)
        with torch.no_grad():
            linear_baseline.weight.data.copy_(
                torch.randint(-1, 2, linear_baseline.weight.shape, device="cuda").to(torch.bfloat16)
            )
        if tp_size > 1:
            sync_linear_weight(linear_baseline, tp_group)
            chunk_in = in_features // tp_size
            linear_tp.weight.data.copy_(
                linear_baseline.weight.data[:, rank * chunk_in:(rank + 1) * chunk_in]
            )
            linear_tp.weight.main_grad = torch.zeros_like(
                linear_tp.weight.data, dtype=torch.float32
            )

        # 整数初始化 input，broadcast 保证一致
        inp_full = torch.randint(-1, 2, (batch_size, in_features), device="cuda").to(torch.bfloat16)
        if tp_size > 1:
            broadcast_tensor(inp_full, tp_group)

        inp_baseline = inp_full.clone().requires_grad_(True)

        # row parallel input: 按 in_features 维度切分 [m, k//tp]
        if tp_size > 1:
            chunk_in = in_features // tp_size
            inp_tp = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous().requires_grad_(True)
        else:
            inp_tp = inp_full.clone().requires_grad_(True)

        # 显式初始化 dy（整数值），broadcast 保证两边一致
        # dy_full shape: [batch, out_features]，与 baseline output shape 一致
        dy_full = torch.randint(-1, 2, (batch_size, out_features), device="cuda").to(torch.bfloat16)
        if tp_size > 1:
            broadcast_tensor(dy_full, tp_group)

        # baseline dy: 完整 [batch, out_features]
        # tp dy: row parallel 输出经过 AllReduce/ReduceScatter
        #   非SP: out_tp shape [batch, out_features]（AllReduce全量）  -> dy_tp = dy_full
        #   SP  : out_tp shape [batch//tp, out_features]（ReduceScatter）-> dy_tp = dy_full[seq_shard, :]
        if sequence_parallel and tp_size > 1:
            chunk_seq = batch_size // tp_size
            dy_tp = dy_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
        else:
            dy_tp = dy_full.contiguous()

        # Forward + backward
        # baseline: inp_full [m, k] @ W_full.T [k, n] -> AllReduce -> [m, n]
        # tp:       inp_tp   [m, k//tp] @ W_r.T [k//tp, n] -> AllReduce/RS -> output
        out_baseline = linear_baseline(inp_baseline, is_first_microbatch=True)
        out_tp       = linear_tp(inp_tp, is_first_microbatch=True)

        out_baseline.backward(dy_full)
        out_tp.backward(dy_tp)

        if rank < tp_size:
            chunk_in = in_features // tp_size

            # --- dgrad 验证 ---
            # row parallel dgrad: dx_r = dy * W_r.T，shape [m, k//tp]
            # 与 baseline dgrad 的对应 in_features 列切片比较
            if inp_baseline.grad is not None and inp_tp.grad is not None:
                baseline_dgrad = inp_baseline.grad[:, rank * chunk_in:(rank + 1) * chunk_in]

                abs_diff = (baseline_dgrad - inp_tp.grad).abs()
                print(f"Rank {rank} - dgrad:")
                print(f"  baseline_dgrad[:, in_shard] shape: {baseline_dgrad.shape}")
                print(f"  inp_tp.grad shape:                 {inp_tp.grad.shape}")
                print(f"  Max absolute diff:                 {abs_diff.max().item():.6e}")
                print(f"  Mean absolute diff:                {abs_diff.mean().item():.6e}")

                torch.testing.assert_close(
                    baseline_dgrad, inp_tp.grad,
                    rtol=0, atol=0,
                    msg=f"Row parallel dgrad mismatch at rank {rank}"
                )

            # --- wgrad 验证 ---
            # row parallel wgrad: dw_r = dy.T * x_r，shape [n, k//tp]
            # 与 baseline wgrad 的对应 in_features 列切片比较
            if linear_baseline.weight.main_grad is not None and linear_tp.weight.main_grad is not None:
                baseline_wgrad = linear_baseline.weight.main_grad[
                    :, rank * chunk_in:(rank + 1) * chunk_in
                ]

                abs_diff = (baseline_wgrad - linear_tp.weight.main_grad).abs()
                print(f"Rank {rank} - wgrad:")
                print(f"  baseline_wgrad[:, in_shard] shape: {baseline_wgrad.shape}")
                print(f"  tp wgrad shape:                    {linear_tp.weight.main_grad.shape}")
                print(f"  Max absolute diff:                 {abs_diff.max().item():.6e}")
                print(f"  Mean absolute diff:                {abs_diff.mean().item():.6e}")

                torch.testing.assert_close(
                    baseline_wgrad, linear_tp.weight.main_grad,
                    rtol=0, atol=0,
                    msg=f"Row parallel wgrad mismatch at rank {rank}"
                )

            print(f"✓ Row parallel backward pass accuracy verified!")

        print(f"{'='*60}\n")

def run_tp_tests(qparam_id: int = 5, tp_size: int = 2, sequence_parallel: bool = True):
    """
    Run TP tests.

    Note: This requires running with torchrun:
    torchrun --nproc_per_node=2 test_linear_tp.py --qparam 5 --tp-size 2 --sp
    torchrun --nproc_per_node=2 test_linear_tp.py --qparam 5 --tp-size 2 --no-sp
    """
    qparam_map = {
        1: QuantizeRecipe.NON_QUANTIZE,
        2: QuantizeRecipe.INT4_SUB_CHANNEL,
        3: QuantizeRecipe.FP4_SUB_CHANNEL_REF,
        4: QuantizeRecipe.FP4_SUB_CHANNEL,
        5: QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET,
    }
    qparam = qparam_map[qparam_id]

    rank = dist.get_rank() if dist.is_initialized() else 0

    if rank == 0:
        print("\n" + "="*60)
        print("Linear Tensor Parallel Accuracy Tests")
        print("="*60)
        print(f"Quantization:      {qparam}")
        print(f"TP Size:           {tp_size}")
        print(f"Sequence Parallel: {sequence_parallel}")
        print("="*60)

    test = TestLinearTP()

    try:
        # Test 1: Column parallel forward
        test.test_column_parallel_forward(qparam, tp_size, sequence_parallel)

        # Test 2: Row parallel forward
        test.test_row_parallel_forward(qparam, tp_size, sequence_parallel)

        # Test 3: Column parallel backward
        test.test_column_parallel_backward(qparam, tp_size, sequence_parallel)

        # Test 4: Row parallel backward
        test.test_row_parallel_backward(qparam, tp_size, sequence_parallel)

        if rank == 0:
            print("\n" + "="*60)
            print("✓ All TP tests passed!")
            print("="*60 + "\n")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Test failed at rank {rank}")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print("\nFull Traceback:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60 + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear TP Accuracy Tests")

    parser.add_argument(
        "--qparam",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="Quantization recipe: 1=NON, 2=INT4, 3=FP4_REF, 4=FP4, 5=FP4_QUARTET (default: 5)"
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=2,
        help="Tensor parallel size (default: 2)"
    )

    parser.add_argument(
        "--sp",
        dest="sequence_parallel",
        action="store_true",
        default=True,
        help="Enable sequence parallel (default: True)"
    )
    parser.add_argument(
        "--no-sp",
        dest="sequence_parallel",
        action="store_false",
        help="Disable sequence parallel"
    )

    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank (set by torchrun)"
    )

    args = parser.parse_args()

    # Initialize distributed
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        initialize_distributed(rank, world_size)
    else:
        print("ERROR: This test requires running with torchrun!")
        print("Usage:")
        print("  # With sequence parallel (default):")
        print("  torchrun --nproc_per_node=2 test_linear_tp.py --qparam 5 --tp-size 2")
        print("  # Without sequence parallel:")
        print("  torchrun --nproc_per_node=2 test_linear_tp.py --qparam 5 --tp-size 2 --no-sp")
        exit(1)

    try:
        run_tp_tests(qparam_id=args.qparam, tp_size=args.tp_size, sequence_parallel=args.sequence_parallel)
    finally:
        cleanup_distributed()
