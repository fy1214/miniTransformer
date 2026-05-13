import os
import torch
import torch.distributed as dist

from miniTransformer.distributed.cutedsl.nvshmem_communicator import (
    init_nvshmem_with_dist,
    finalize_nvshmem,
)
from miniTransformer.distributed.cutedsl.gemm_allreduce import GemmAllReduceCtx


# ── NVTX helper ───────────────────────────────────────────────────────────────

try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False


class NvtxRange:
    """nvtx range context manager，无 nvtx 时 no-op。"""
    def __init__(self, name: str, color: str = "blue"):
        self.name  = name
        self.color = color

    def __enter__(self):
        if HAS_NVTX:
            nvtx.push_range(self.name, color=self.color)
        return self

    def __exit__(self, *args):
        if HAS_NVTX:
            nvtx.pop_range()


# ── 三种方案 ──────────────────────────────────────────────────────────────────

def scheme_a(a_local, b_local, tp_group):
    out = (a_local @ b_local.T).clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out


def scheme_b(a_local, b_local, tp_group):
    out = (a_local @ b_local.T).float()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=tp_group)
    return out.to(torch.bfloat16)


def scheme_c(a_local, b_local, ctx):
    return ctx.forward(a_local, b_local)


# ── ncu 模式：单进程，不需要 dist ─────────────────────────────────────────────

def run_ncu_mode(args):
    """
    ncu 模式：单进程运行 CuTeDSL kernel，供 ncu attach 抓指标。
    不使用 dist AllReduce（ncu 不支持多进程），只测量 kernel 本身。
    """
    torch.cuda.set_device(0)

    M, N, K = args.M, args.N, args.K
    print(f"[ncu mode] M={M} N={N} K={K}  warmup={args.warmup}  iters={args.iters}")
    print(f"[ncu mode] nvshmem 未初始化，仅测量 kernel（无 AllReduce）")

    # 对 ncu 模式，我们直接测 bf16 GEMM，不需要 nvshmem
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    # warmup
    for _ in range(args.warmup):
        _ = a @ b.T
    torch.cuda.synchronize()

    # 正式 profile（ncu attach 到这里）
    print(f"[ncu mode] 开始 profile（{args.iters} 次）...")
    for i in range(args.iters):
        with NvtxRange(f"bf16_gemm_iter{i}", color="green"):
            out = a @ b.T
            torch.cuda.synchronize()

    print(f"[ncu mode] 完成")


# ── nsys 模式：多进程，抓时间线 ──────────────────────────────────────────────

def run_nsys_mode(args):
    """
    nsys 模式：多进程，用 nvtx range 标记三种方案，供 nsys 时间线分析。
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dev = init_nvshmem_with_dist()

    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    tp_group   = dist.new_group(ranks=list(range(world_size)))

    M, N, K = args.M, args.N, args.K
    chunk   = K // world_size

    if rank == 0:
        print(f"[nsys mode] M={M} N={N} K={K}  tp={world_size}")
        print(f"[nsys mode] warmup={args.warmup}  iters={args.iters}")

    # 生成输入
    torch.manual_seed(42)
    a_full = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b_full = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    dist.broadcast(a_full, src=0, group=tp_group)
    dist.broadcast(b_full, src=0, group=tp_group)
    a_local = a_full[:, rank * chunk:(rank + 1) * chunk].contiguous()
    b_local = b_full[:, rank * chunk:(rank + 1) * chunk].contiguous()

    # 初始化 CuTeDSL ctx
    ctx = GemmAllReduceCtx(tp_group=tp_group)
    _ = ctx.forward(a_local, b_local)   # compile
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)

    # ── Warmup ────────────────────────────────────────────────────────────────
    for _ in range(args.warmup):
        scheme_a(a_local, b_local, tp_group)
        scheme_b(a_local, b_local, tp_group)
        scheme_c(a_local, b_local, ctx)
    torch.cuda.synchronize()
    dist.barrier(group=tp_group)

    if rank == 0:
        print(f"[nsys mode] warmup 完成，开始 profile...")

    # ── 正式 profile（nsys 抓这里的时间线）───────────────────────────────────
    for i in range(args.iters):
        # 方案 A：bf16 GEMM + bf16 AllReduce
        with NvtxRange(f"A_bf16_gemm_bf16_ar_iter{i}", color="red"):
            scheme_a(a_local, b_local, tp_group)
        torch.cuda.synchronize()

        # 方案 B：bf16 GEMM + fp32 AllReduce
        with NvtxRange(f"B_bf16_gemm_fp32_ar_iter{i}", color="yellow"):
            scheme_b(a_local, b_local, tp_group)
        torch.cuda.synchronize()

        # 方案 C：CuTeDSL tile-level fp32 AllReduce
        with NvtxRange(f"C_cutedsl_fp32_ar_iter{i}", color="green"):
            scheme_c(a_local, b_local, ctx)
        torch.cuda.synchronize()

        dist.barrier(group=tp_group)

    if rank == 0:
        print(f"[nsys mode] profile 完成")

    ctx.free()
    finalize_nvshmem()
    dist.destroy_process_group()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GEMM AllReduce Profile")
    parser.add_argument("--mode",   choices=["ncu", "nsys"], default="nsys")
    parser.add_argument("--M",      type=int, default=2048)
    parser.add_argument("--N",      type=int, default=2048)
    parser.add_argument("--K",      type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=3,
                        help="warmup 次数")
    parser.add_argument("--iters",  type=int, default=5,
                        help="profile 次数")
    args = parser.parse_args()

    if args.mode == "ncu":
        run_ncu_mode(args)
    else:
        if "RANK" not in os.environ:
            print("nsys 模式需要 torchrun 启动，见文件顶部的使用说明")
            raise SystemExit(1)
        run_nsys_mode(args)


if __name__ == "__main__":
    main()