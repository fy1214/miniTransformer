"""
TE Linear Tensor Parallel (TP) 通信正确性测试
=============================================

验证目标：TE Linear 在 TP 下的通信逻辑（AllGather/ReduceScatter/AllReduce）是否正确。

策略：
  - 不使用任何量化 recipe，直接用 bf16 forward
  - 输入 x、weight、grad 均初始化为整数值 {-1, 0, 1}
    → 乘积和累加均为整数，bf16 可精确表示，排除浮点误差干扰
  - bit-exact 验证（rtol=0, atol=0）

测试矩阵：
  test_column_parallel_forward
  test_column_parallel_backward_dgrad
  test_column_parallel_backward_wgrad
  test_row_parallel_forward
  test_row_parallel_backward_dgrad
  test_row_parallel_backward_wgrad

运行：
  torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py
  torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py --no-sp
"""

import argparse
import os
import traceback

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te


# ==================== Distributed Utilities ====================

def initialize_distributed():
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://",
        )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_tp_group(tp_size: int):
    world_size = dist.get_world_size()
    if tp_size > world_size:
        raise ValueError(f"tp_size ({tp_size}) > world_size ({world_size})")
    tp_group = dist.new_group(ranks=list(range(tp_size)))
    return tp_group if dist.get_rank() < tp_size else None


def broadcast_weight(linear, tp_group):
    """rank 0 的 weight broadcast 到所有 rank。"""
    dist.broadcast(linear.weight.data, src=0, group=tp_group)
    if hasattr(linear, "bias") and linear.bias is not None and linear.bias.numel() > 0:
        dist.broadcast(linear.bias.data, src=0, group=tp_group)


def broadcast_tensor(t: torch.Tensor, tp_group) -> torch.Tensor:
    dist.broadcast(t, src=0, group=tp_group)
    return t


def randint_bf16(*shape, device="cuda"):
    """生成 {-1, 0, 1} 整数值的 bfloat16 张量。"""
    return torch.randint(-1, 2, shape, device=device).to(torch.bfloat16)


# ==================== Module Builders ====================

def build_baseline(in_features, out_features, parallel_mode, bias, seed) -> te.Linear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        params_dtype=torch.bfloat16,
        parallel_mode=parallel_mode,
        tp_size=1,
        device="cuda",
    )


def build_tp(in_features, out_features, parallel_mode, bias,
             tp_size, tp_group, sequence_parallel, seed) -> te.Linear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        params_dtype=torch.bfloat16,
        parallel_mode=parallel_mode,
        tp_size=tp_size,
        tp_group=tp_group,
        sequence_parallel=sequence_parallel,
        device="cuda",
    )


# ==================== Weight Shard Copy ====================

def copy_column_shard(linear_tp, linear_baseline, rank, tp_size):
    """Column parallel weight[out/tp, in]：从 baseline 行切片。"""
    chunk = linear_baseline.weight.shape[0] // tp_size
    shard = linear_baseline.weight.data[rank * chunk:(rank + 1) * chunk, :].contiguous().clone()
    linear_tp.weight = torch.nn.Parameter(shard)


def copy_row_shard(linear_tp, linear_baseline, rank, tp_size):
    """Row parallel weight[out, in/tp]：从 baseline 列切片。"""
    chunk = linear_baseline.weight.shape[1] // tp_size
    shard = linear_baseline.weight.data[:, rank * chunk:(rank + 1) * chunk].contiguous().clone()
    linear_tp.weight = torch.nn.Parameter(shard)


# ==================== Shared Setup Helper ====================

def _setup_column(tp_size, sequence_parallel, in_features, out_features,
                  total_batch, bias, seed, tp_group):
    rank = dist.get_rank()

    # 构建模块
    baseline = build_baseline(in_features, out_features, "column", bias, seed)
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)

    tp_linear = build_tp(in_features, out_features, "column", bias,
                         tp_size, tp_group, sequence_parallel, seed)
    if tp_size > 1:
        copy_column_shard(tp_linear, baseline, rank, tp_size)

    # 整数初始化 weight（覆盖随机初始化）
    torch.manual_seed(seed + 10)
    with torch.no_grad():
        baseline.weight.data.copy_(randint_bf16(*baseline.weight.shape))
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)
        copy_column_shard(tp_linear, baseline, rank, tp_size)

    # 整数初始化 input，broadcast 保证一致
    torch.manual_seed(seed + 100)
    inp_full = randint_bf16(total_batch, in_features)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    # 整数初始化 grad，broadcast 保证一致
    torch.manual_seed(seed + 200)
    grad_full = randint_bf16(total_batch, out_features)
    if tp_size > 1:
        broadcast_tensor(grad_full, tp_group)

    chunk_out = out_features // tp_size

    # inp for TP: SP 时取 seq shard，非 SP 时用完整 input
    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        inp_tp_data = inp_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
    else:
        inp_tp_data = inp_full

    # grad for TP: column parallel out_tp shape 始终为 [total_batch, out//tp]
    grad_tp = grad_full[:, rank * chunk_out:(rank + 1) * chunk_out].contiguous()

    # grad for baseline: out_baseline shape [total_batch, out_features]
    # 只激活 col_shard 列，其余置 0，保证与 tp dgrad 数学等价
    grad_baseline = torch.zeros_like(grad_full)
    grad_baseline[:, rank * chunk_out:(rank + 1) * chunk_out] = \
        grad_full[:, rank * chunk_out:(rank + 1) * chunk_out]

    return dict(
        baseline=baseline, tp_linear=tp_linear,
        inp_full=inp_full, inp_tp_data=inp_tp_data,
        grad_full=grad_full, grad_tp=grad_tp, grad_baseline=grad_baseline,
        chunk_out=chunk_out,
    )


def _setup_row(tp_size, sequence_parallel, in_features, out_features,
               total_batch, bias, seed, tp_group):
    rank = dist.get_rank()

    # 构建模块
    baseline = build_baseline(in_features, out_features, "row", bias, seed)
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)

    tp_linear = build_tp(in_features, out_features, "row", bias,
                         tp_size, tp_group, sequence_parallel, seed)
    if tp_size > 1:
        copy_row_shard(tp_linear, baseline, rank, tp_size)

    # 整数初始化 weight（覆盖随机初始化）
    torch.manual_seed(seed + 10)
    with torch.no_grad():
        baseline.weight.data.copy_(randint_bf16(*baseline.weight.shape))
    if tp_size > 1:
        broadcast_weight(baseline, tp_group)
        copy_row_shard(tp_linear, baseline, rank, tp_size)

    # 整数初始化 input，broadcast 保证一致
    torch.manual_seed(seed + 100)
    inp_full = randint_bf16(total_batch, in_features)
    if tp_size > 1:
        broadcast_tensor(inp_full, tp_group)

    # 整数初始化 grad，broadcast 保证一致
    torch.manual_seed(seed + 200)
    grad_full = randint_bf16(total_batch, out_features)
    if tp_size > 1:
        broadcast_tensor(grad_full, tp_group)

    chunk_in = in_features // tp_size

    # inp for TP: row parallel 始终按 in_features 维度切分
    if tp_size > 1:
        inp_tp_data = inp_full[:, rank * chunk_in:(rank + 1) * chunk_in].contiguous()
    else:
        inp_tp_data = inp_full

    # grad for TP output
    # SP=True : out_tp shape [total_batch//tp, out_features]（ReduceScatter）
    # SP=False: out_tp shape [total_batch, out_features]（AllReduce）
    if sequence_parallel and tp_size > 1:
        chunk_seq = total_batch // tp_size
        grad_tp = grad_full[rank * chunk_seq:(rank + 1) * chunk_seq, :].contiguous()
    else:
        grad_tp = grad_full.contiguous()

    return dict(
        baseline=baseline, tp_linear=tp_linear,
        inp_full=inp_full, inp_tp_data=inp_tp_data,
        grad_full=grad_full, grad_tp=grad_tp,
        chunk_in=chunk_in,
    )


# ==================== Column Parallel Tests ====================

def test_column_parallel_forward(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
    **kwargs,
):
    """
    Column parallel forward 通信正确性验证（bit-exact）。

    SP=True : inp_tp = inp_full[seq_shard] → AllGather → GEMM → out_tp [total_batch, out//tp]
              compare out_tp vs out_baseline[:, col_shard]
    SP=False: inp_tp = inp_full → GEMM → out_tp [total_batch, out//tp]
              compare out_tp vs out_baseline[:, col_shard]
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_column(tp_size, sequence_parallel, in_features, out_features,
                      total_batch, bias, seed, tp_group)

    with torch.no_grad():
        out_baseline = s["baseline"](s["inp_full"].clone(), is_first_microbatch=True)
        out_tp = s["tp_linear"](s["inp_tp_data"].clone(), is_first_microbatch=True)

    if rank < tp_size:
        chunk_out = s["chunk_out"]
        ref = out_baseline[:, rank * chunk_out:(rank + 1) * chunk_out]

        diff = (ref - out_tp).abs()
        print(f"  [col fwd] rank={rank} out_tp={out_tp.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, out_tp, rtol=0, atol=0,
                                   msg=f"[col fwd] rank {rank} mismatch")
        print(f"  ✓ test_column_parallel_forward rank={rank} passed")


def test_column_parallel_backward_dgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
    **kwargs,
):
    """
    Column parallel dgrad 通信正确性验证（bit-exact）。

    SP=True : dgrad ReduceScattered → inp_tp.grad [total_batch//tp, in]
              compare vs inp_baseline.grad[seq_shard, :]
    SP=False: dgrad AllReduced → inp_tp.grad [total_batch, in]
              compare vs inp_baseline.grad
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_column(tp_size, sequence_parallel, in_features, out_features,
                      total_batch, bias, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_baseline"])

    out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size and inp_baseline.grad is not None and inp_tp.grad is not None:
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = inp_baseline.grad[rank * chunk_seq:(rank + 1) * chunk_seq, :]
        else:
            ref = inp_baseline.grad

        diff = (ref - inp_tp.grad).abs()
        print(f"  [col bwd dgrad] rank={rank} grad_shape={inp_tp.grad.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, inp_tp.grad, rtol=0, atol=0,
                                   msg=f"[col bwd dgrad] rank {rank} mismatch")
        print(f"  ✓ test_column_parallel_backward_dgrad rank={rank} passed")


def test_column_parallel_backward_wgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
    **kwargs,
):
    """
    Column parallel wgrad 通信正确性验证（bit-exact）。

    tp_linear.weight.grad shape: [out//tp, in]
    compare vs baseline.weight.grad[col_shard, :]
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_column(tp_size, sequence_parallel, in_features, out_features,
                      total_batch, bias, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_baseline"])

    out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size:
        if s["baseline"].weight.grad is not None and s["tp_linear"].weight.grad is not None:
            chunk_out = s["chunk_out"]
            ref = s["baseline"].weight.grad[rank * chunk_out:(rank + 1) * chunk_out, :]
            diff = (ref - s["tp_linear"].weight.grad).abs()
            print(f"  [col bwd wgrad] rank={rank} wgrad_shape={s['tp_linear'].weight.grad.shape} "
                  f"max={diff.max():.4e} mean={diff.mean():.4e}")
            torch.testing.assert_close(ref, s["tp_linear"].weight.grad, rtol=0, atol=0,
                                       msg=f"[col bwd wgrad] rank {rank} mismatch")
            print(f"  ✓ test_column_parallel_backward_wgrad rank={rank} passed")


# ==================== Row Parallel Tests ====================

def test_row_parallel_forward(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
    **kwargs,
):
    """
    Row parallel forward 通信正确性验证（bit-exact）。

    SP=True : ReduceScatter → out_tp [total_batch//tp, out]
              compare vs out_baseline[seq_shard, :]
    SP=False: AllReduce → out_tp [total_batch, out]
              compare vs out_baseline
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_row(tp_size, sequence_parallel, in_features, out_features,
                   total_batch, bias, seed, tp_group)

    with torch.no_grad():
        out_baseline = s["baseline"](s["inp_full"].clone(), is_first_microbatch=True)
        out_tp = s["tp_linear"](s["inp_tp_data"].clone(), is_first_microbatch=True)

    if rank < tp_size:
        if sequence_parallel and tp_size > 1:
            chunk_seq = total_batch // tp_size
            ref = out_baseline[rank * chunk_seq:(rank + 1) * chunk_seq, :]
        else:
            ref = out_baseline

        diff = (ref - out_tp).abs()
        print(f"  [row fwd] rank={rank} out_tp={out_tp.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, out_tp, rtol=0, atol=0,
                                   msg=f"[row fwd] rank {rank} mismatch")
        print(f"  ✓ test_row_parallel_forward rank={rank} passed")


def test_row_parallel_backward_dgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
    **kwargs,
):
    """
    Row parallel dgrad 通信正确性验证（bit-exact）。

    inp_tp.grad shape: [total_batch, in//tp]
    compare vs inp_baseline.grad[:, in_shard]
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_row(tp_size, sequence_parallel, in_features, out_features,
                   total_batch, bias, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_full"])

    out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size and inp_baseline.grad is not None and inp_tp.grad is not None:
        chunk_in = s["chunk_in"]
        ref = inp_baseline.grad[:, rank * chunk_in:(rank + 1) * chunk_in]
        diff = (ref - inp_tp.grad).abs()
        print(f"  [row bwd dgrad] rank={rank} grad_shape={inp_tp.grad.shape} "
              f"max={diff.max():.4e} mean={diff.mean():.4e}")
        torch.testing.assert_close(ref, inp_tp.grad, rtol=0, atol=0,
                                   msg=f"[row bwd dgrad] rank {rank} mismatch")
        print(f"  ✓ test_row_parallel_backward_dgrad rank={rank} passed")


def test_row_parallel_backward_wgrad(
    tp_size: int, sequence_parallel: bool,
    in_features: int = 512, out_features: int = 512,
    batch_size: int = 32, seq_len: int = 32,
    bias: bool = False, seed: int = 42,
    **kwargs,
):
    """
    Row parallel wgrad 通信正确性验证（bit-exact）。

    tp_linear.weight.grad shape: [out, in//tp]
    compare vs baseline.weight.grad[:, in_shard]
    """
    tp_group = create_tp_group(tp_size) if tp_size > 1 else None
    rank = dist.get_rank()
    total_batch = batch_size * seq_len

    s = _setup_row(tp_size, sequence_parallel, in_features, out_features,
                   total_batch, bias, seed, tp_group)

    inp_baseline = s["inp_full"].clone().requires_grad_(True)
    inp_tp = s["inp_tp_data"].clone().requires_grad_(True)

    out_baseline = s["baseline"](inp_baseline, is_first_microbatch=True)
    out_baseline.backward(s["grad_full"])

    out_tp = s["tp_linear"](inp_tp, is_first_microbatch=True)
    out_tp.backward(s["grad_tp"])

    if rank < tp_size:
        if s["baseline"].weight.grad is not None and s["tp_linear"].weight.grad is not None:
            chunk_in = s["chunk_in"]
            ref = s["baseline"].weight.grad[:, rank * chunk_in:(rank + 1) * chunk_in]
            diff = (ref - s["tp_linear"].weight.grad).abs()
            print(f"  [row bwd wgrad] rank={rank} wgrad_shape={s['tp_linear'].weight.grad.shape} "
                  f"max={diff.max():.4e} mean={diff.mean():.4e}")
            torch.testing.assert_close(ref, s["tp_linear"].weight.grad, rtol=0, atol=0,
                                       msg=f"[row bwd wgrad] rank {rank} mismatch")
            print(f"  ✓ test_row_parallel_backward_wgrad rank={rank} passed")


# ==================== Main ====================

ALL_TESTS = [
    ("Column Parallel Forward",        test_column_parallel_forward),
    ("Column Parallel Backward dgrad", test_column_parallel_backward_dgrad),
    ("Column Parallel Backward wgrad", test_column_parallel_backward_wgrad),
    ("Row Parallel Forward",           test_row_parallel_forward),
    ("Row Parallel Backward dgrad",    test_row_parallel_backward_dgrad),
    ("Row Parallel Backward wgrad",    test_row_parallel_backward_wgrad),
]


def run_tests(tp_size: int = 2, sequence_parallel: bool = True):
    rank = dist.get_rank()

    if rank == 0:
        print("\n" + "=" * 60)
        print("TE Linear TP 通信正确性测试（bf16 整数输入，bit-exact）")
        print("=" * 60)
        print(f"  tp_size:           {tp_size}")
        print(f"  sequence_parallel: {sequence_parallel}")
        print("=" * 60)

    for i, (name, fn) in enumerate(ALL_TESTS, 1):
        if rank == 0:
            print(f"\n[{i}/{len(ALL_TESTS)}] {name}")
        try:
            fn(tp_size=tp_size, sequence_parallel=sequence_parallel)
            dist.barrier()
        except Exception as e:
            print(f"\n  ✗ FAILED at rank {rank}: {e}")
            traceback.print_exc()
            raise

    if rank == 0:
        print("\n" + "=" * 60)
        print(f"✓ 全部 {len(ALL_TESTS)} 项测试通过！")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--sp",    dest="sequence_parallel", action="store_true",  default=True)
    parser.add_argument("--no-sp", dest="sequence_parallel", action="store_false")
    args = parser.parse_args()

    if "RANK" not in os.environ:
        print("请用 torchrun 启动：")
        print("  torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py")
        print("  torchrun --nproc_per_node=2 test_te_linear_nvfp4_tp.py --no-sp")
        raise SystemExit(1)

    initialize_distributed()
    try:
        run_tests(tp_size=args.tp_size, sequence_parallel=args.sequence_parallel)
    finally:
        cleanup_distributed()
