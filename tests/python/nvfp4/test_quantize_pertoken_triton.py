import torch
import triton
import numpy as np

from miniTransformer.ops.triton.nvfp4_quantization import quant_nvfp4_pertoken_golden, quant_nvfp4_pertoken_triton


# ============================================================================
# 精度对比脚本
# ============================================================================

def compare_all_outputs():
    """全面对比golden和triton的所有输出"""

    test_configs = [
        (32, 128, 16, 42),
        (128, 256, 16, 123),
        (64, 512, 16, 456),
        (256, 1024, 16, 789),
    ]

    print("=" * 100)
    print("Golden vs Triton 精度对比")
    print("=" * 100)

    all_pass = True

    for M, K, block_size, seed in test_configs:
        print(f"\n{'=' * 80}")
        print(f"测试配置: M={M}, K={K}, block_size={block_size}, seed={seed}")
        print(f"{'=' * 80}")

        torch.manual_seed(seed)
        x = torch.randn(M, K, device='cuda')

        x_q_golden, gs_golden, bs_golden = quant_nvfp4_pertoken_golden(x, block_size)
        x_q_triton, gs_triton, bs_triton = quant_nvfp4_pertoken_triton(x, block_size)

        B = K // block_size

        # Global Scale
        gs_diff = (gs_golden - gs_triton).abs()
        gs_max_diff = gs_diff.max().item()
        gs_mean_diff = gs_diff.mean().item()
        gs_exact = (gs_diff < 1e-7).float().mean().item() * 100

        print(f"\n[Global Scale] shape={gs_golden.shape}")
        print(f"  Golden: [{gs_golden.min().item():.6e}, {gs_golden.max().item():.6e}]")
        print(f"  Triton: [{gs_triton.min().item():.6e}, {gs_triton.max().item():.6e}]")
        print(f"  Max diff:  {gs_max_diff:.6e}")
        print(f"  Mean diff: {gs_mean_diff:.6e}")
        print(f"  Exact:     {gs_exact:.2f}%")
        print(f"  Status:    {'✓' if gs_max_diff < 1e-6 else '✗'}")

        if gs_max_diff >= 1e-5:
            all_pass = False

        # Block Scale
        bs_diff = (bs_golden - bs_triton).abs()
        bs_max_diff = bs_diff.max().item()
        bs_mean_diff = bs_diff.mean().item()
        bs_exact = (bs_diff < 1e-7).float().mean().item() * 100

        print(f"\n[Block Scale] shape={bs_golden.shape}")
        print(f"  Max diff:  {bs_max_diff:.6e}")
        print(f"  Mean diff: {bs_mean_diff:.6e}")
        print(f"  Exact:     {bs_exact:.2f}%")
        print(f"  Status:    {'✓' if bs_max_diff < 1e-6 else '✗'}")

        if bs_max_diff >= 1e-5:
            all_pass = False

        # x_q
        xq_diff = (x_q_golden - x_q_triton).abs()
        xq_max_diff = xq_diff.max().item()
        xq_mean_diff = xq_diff.mean().item()
        xq_exact = (xq_diff < 1e-7).float().mean().item() * 100

        print(f"\n[x_q] shape={x_q_golden.shape}")
        print(f"  Max diff:  {xq_max_diff:.6e}")
        print(f"  Mean diff: {xq_mean_diff:.6e}")
        print(f"  Exact:     {xq_exact:.2f}%")
        print(f"  Status:    {'✓' if xq_max_diff < 1e-6 else '✗'}")

        if xq_max_diff >= 1e-5:
            all_pass = False
            mismatches = (xq_diff > 1e-5).sum().item()
            print(f"  Mismatches: {mismatches}")

        print(
            f"  -> GS:{'✓' if gs_max_diff < 1e-5 else '✗'} BS:{'✓' if bs_max_diff < 1e-5 else '✗'} x_q:{'✓' if xq_max_diff < 1e-5 else '✗'}")

    print(f"\n{'=' * 100}")
    print(f"结果: {'✓ 全部通过' if all_pass else '✗ 存在差异'}")
    print(f"{'=' * 100}")

    return all_pass

# ============================================================================
# 性能测试
# ============================================================================

def benchmark():
    """性能对比"""
    import time

    shapes = [
        (32, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
        (1024, 4096),
        (4096, 4096),
    ]

    print("\n" + "=" * 80)
    print(f"{'Shape':<20} {'Golden (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("-" * 80)

    for M, K in shapes:
        x = torch.randn(M, K, device='cuda')

        # Warmup
        for _ in range(10):
            quant_nvfp4_pertoken_golden(x)
            quant_nvfp4_pertoken_triton(x)
        torch.cuda.synchronize()

        # Golden
        t1 = time.perf_counter()
        for _ in range(100):
            quant_nvfp4_pertoken_golden(x)
        torch.cuda.synchronize()
        t1 = (time.perf_counter() - t1) / 100 * 1000

        # Triton
        t2 = time.perf_counter()
        for _ in range(100):
            quant_nvfp4_pertoken_triton(x)
        torch.cuda.synchronize()
        t2 = (time.perf_counter() - t2) / 100 * 1000

        speedup = t1 / t2
        print(f"{f'{M}x{K}':<20} {t1:<15.3f} {t2:<15.3f} {speedup:<10.2f}x")


if __name__ == "__main__":
    compare_all_outputs()