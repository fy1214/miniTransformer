"""Benchmark NVFP4 grouped pertoken: legacy per-row scheduler vs chunked (CHUNK_M=128
+ binary search) scheduler. Both consume already-padded inputs (m_per_group %
128 == 0), so this isolates the scheduling-strategy delta from any pad-fusion
effect.

Routing model: Qwen3-30B-A3B (G=128, top_k=8).
"""

import torch
import fp4_gemm

NVFP4_DATA  = torch.uint8
NVFP4_SCALE = torch.float8_e4m3fn
PAD = 128
SCALE_OVERRIDE = 1.0


def build_offsets(m_splits, K):
    device = 'cuda'
    row = [0]
    sf  = [0]
    for m in m_splits:
        row.append(row[-1] + m)
        sf .append(sf [-1] + m * (K // 16))
    return (
        torch.tensor(row, dtype=torch.int32, device=device),
        torch.tensor(sf,  dtype=torch.int64, device=device),
    )


def cuda_time(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # ms -> us


def equiv_check():
    """Both kernels must produce identical FP4 + SF + global_scale on
    128-aligned inputs."""
    print("== equivalence check (legacy per-row vs chunked) ==")
    for G, m, K in [
        (8,   128, 2048),
        (16,  128, 768),
        (32,  256, 2048),
        (128, 128, 2048),
        (62,  128, 768),
    ]:
        m_splits = [m] * G
        total = G * m
        row_off, sf_off = build_offsets(m_splits, K)
        torch.manual_seed(0)
        x = torch.randn(total, K, dtype=torch.bfloat16, device='cuda')

        sf_bytes = total * (K // 16)
        out_a = torch.empty((total, K // 2), dtype=NVFP4_DATA,  device='cuda')
        sf_a  = torch.zeros(sf_bytes,        dtype=NVFP4_SCALE, device='cuda')
        gs_a  = torch.empty(total,           dtype=torch.float32, device='cuda')
        fp4_gemm.rtn_fp4_group_pertoken_legacy_per_row(
            out_a, sf_a, gs_a, x, row_off, sf_off, SCALE_OVERRIDE)

        out_b = torch.empty_like(out_a)
        sf_b  = torch.zeros_like(sf_a)
        gs_b  = torch.empty_like(gs_a)
        fp4_gemm.rtn_fp4_group_pertoken(
            out_b, sf_b, gs_b, x, row_off, sf_off, SCALE_OVERRIDE)

        ok_fp4 = torch.equal(out_a, out_b)
        ok_sf  = torch.equal(sf_a, sf_b)
        ok_gs  = torch.equal(gs_a, gs_b)
        tag = "OK" if (ok_fp4 and ok_sf and ok_gs) else "FAIL"
        print(f"  [{tag}]  G={G:>3} m={m:>4} K={K:>5}   fp4={ok_fp4} sf={ok_sf} gs={ok_gs}")
        if not (ok_fp4 and ok_sf and ok_gs):
            diff_fp4 = (out_a.view(torch.uint8) != out_b.view(torch.uint8)).sum().item()
            diff_sf  = (sf_a.view(torch.uint8) != sf_b.view(torch.uint8)).sum().item()
            diff_gs  = (gs_a != gs_b).sum().item()
            print(f"          diff: fp4={diff_fp4}/{out_a.numel()}"
                  f"  sf={diff_sf}/{sf_a.numel()}  gs={diff_gs}/{gs_a.numel()}")


# Qwen3-30B-A3B routing shapes; balanced routing → m_per_group = tokens*8/G.
# Skipping tokens=1024 because m=64 is not 128-aligned and the chunked kernel
# only accepts padded-aligned inputs (post-Fp8Padding is the contract).
QWEN3_CONFIGS = [
    # tokens,  K
    (2048,   2048),   # gate_up, m=128
    (4096,   2048),   # m=256
    (8192,   2048),   # m=512
    (16384,  2048),   # m=1024
    (32768,  2048),   # m=2048
    (2048,    768),   # down, m=128
    (4096,    768),   # m=256
    (8192,    768),   # m=512
    (16384,   768),   # m=1024
    (32768,   768),   # m=2048
]


def bench():
    G = 128
    print(f"\n{'='*108}")
    print(f"NVFP4 grouped pertoken — legacy per-row vs chunked  "
          f"G={G}  [{torch.cuda.get_device_name(0)}]")
    print(f"{'='*108}")
    print(f"{'tokens':>6} {'layer':>8} {'K':>5} {'m':>5} | "
          f"{'legacy us':>10} {'chunked us':>11} | "
          f"{'speedup':>8}")
    print(f"{'-'*108}")

    for tokens, K in QWEN3_CONFIGS:
        total = tokens * 8
        m = total // G
        assert m % PAD == 0, f"chunked requires 128-aligned m (got {m})"
        layer = "gate_up" if K == 2048 else "down"

        m_splits = [m] * G
        row_off, sf_off = build_offsets(m_splits, K)
        x = torch.randn(total, K, dtype=torch.bfloat16, device='cuda')

        sf_bytes = total * (K // 16)
        out_a = torch.empty((total, K // 2), dtype=NVFP4_DATA,  device='cuda')
        sf_a  = torch.zeros(sf_bytes,        dtype=NVFP4_SCALE, device='cuda')
        gs_a  = torch.empty(total,           dtype=torch.float32, device='cuda')
        out_b = torch.empty_like(out_a)
        sf_b  = torch.zeros_like(sf_a)
        gs_b  = torch.empty_like(gs_a)

        def legacy():
            fp4_gemm.rtn_fp4_group_pertoken_legacy_per_row(
                out_a, sf_a, gs_a, x, row_off, sf_off, SCALE_OVERRIDE)

        def chunked():
            fp4_gemm.rtn_fp4_group_pertoken(
                out_b, sf_b, gs_b, x, row_off, sf_off, SCALE_OVERRIDE)

        t_legacy  = cuda_time(legacy)
        t_chunked = cuda_time(chunked)

        print(f"{tokens:>6} {layer:>8} {K:>5} {m:>5} | "
              f"{t_legacy:>10.2f} {t_chunked:>11.2f} | "
              f"{t_legacy/t_chunked:>7.2f}x")
    print(f"{'='*108}")


if __name__ == "__main__":
    equiv_check()
    bench()
