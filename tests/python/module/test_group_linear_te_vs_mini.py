"""
GroupedLinear NVFP4 精度对比测试：miniTransformer vs TransformerEngine
=======================================================================

对比 miniTransformer.GroupedLinear（NVFP4 recipe）与
TransformerEngine.GroupedLinear（NVFP4BlockScaling recipe）的输出精度。

验证策略：
  - 两边使用完全相同的 weight（rank 0 初始化后直接 copy）
  - 两边使用完全相同的 input
  - 对比 forward 输出、dgrad、wgrad 的数值差异
  - 不要求 bit-exact（两边量化实现不同），检查误差在合理范围内

运行：
  python3 test_group_linear_te_vs_mini.py
  python3 test_group_linear_te_vs_mini.py --num-gemms 4 --in-features 512 --out-features 512
"""

import argparse
import traceback

import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.group_linear import GroupedLinear as MiniGroupedLinear


# ==================== Recipe ====================

def make_nvfp4_recipe() -> recipe.NVFP4BlockScaling:
    """标准 NVFP4BlockScaling recipe。"""
    r = recipe.NVFP4BlockScaling()
    r.fp4_quant_fwd_inp    = recipe.QParams()
    r.fp4_quant_fwd_weight = recipe.QParams()
    r.fp4_quant_bwd_grad   = recipe.QParams()
    return r


# ==================== Module Builders ====================

def build_te_grouped_linear(
    num_gemms: int,
    in_features: int,
    out_features: int,
    bias: bool = False,
    seed: int = 42,
) -> te.GroupedLinear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        params_dtype=torch.bfloat16,
        device="cuda",
    )


def build_mini_grouped_linear(
    num_gemms: int,
    in_features: int,
    out_features: int,
    bias: bool = False,
    seed: int = 42,
    qparam: QuantizeRecipe = QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET,
    fuse_wgrad_accumulation: bool = False,
) -> MiniGroupedLinear:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    qlinear_params = get_qlinear_params_from_predefined(qparam)
    return MiniGroupedLinear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        sequence_parallel=False,
        fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        tp_group=None,
        tp_size=1,
        get_rng_state_tracker=None,
        rng_tracker_name=None,
        init_method=None,
        bias=bias,
        return_bias=False,
        params_dtype=torch.bfloat16,
        parallel_mode=None,
        device="cuda",
        ub_overlap_rs=False,
        ub_overlap_ag=False,
        ub_name="test_te_vs_mini",
        layer_number=0,
        qlinear_params=qlinear_params,
    )


# ==================== Weight Sync ====================

def sync_weights_te_to_mini(te_model: te.GroupedLinear,
                              mini_model: MiniGroupedLinear,
                              num_gemms: int) -> None:
    """
    将 TE GroupedLinear 的 weight 复制到 miniTransformer GroupedLinear。
    TE 的参数命名为 weight0, weight1, ..., weightN-1。
    mini 的参数命名相同。
    """
    with torch.no_grad():
        for i in range(num_gemms):
            te_w   = getattr(te_model,   f"weight{i}").data
            mini_w = getattr(mini_model, f"weight{i}").data
            assert te_w.shape == mini_w.shape, \
                f"weight{i} shape mismatch: TE={te_w.shape}, mini={mini_w.shape}"
            mini_w.copy_(te_w)


def init_integer_weights(model, num_gemms: int, seed: int = 42) -> None:
    """将所有权重初始化为整数值 {-1, 0, 1}，消除量化 amax 差异的影响。"""
    torch.manual_seed(seed)
    with torch.no_grad():
        for i in range(num_gemms):
            w = getattr(model, f"weight{i}")
            w.data.copy_(
                torch.randint(-1, 2, w.shape, device=w.device).to(w.dtype)
            )


# ==================== Test Functions ====================

def test_forward(
    num_gemms: int = 4,
    in_features: int = 512,
    out_features: int = 512,
    m_per_expert: int = 64,
    bias: bool = False,
    seed: int = 42,
    qparam: QuantizeRecipe = QuantizeRecipe.FP4_SUB_CHANNEL,
    rtol: float = 0.5,
    atol: float = 0.5,
):
    """
    Forward 精度对比：miniTransformer vs TE。

    使用相同 weight 和 input，对比 forward 输出差异。
    两边量化实现不同（TE 用 NVFP4BlockScaling，mini 用 FP4_SUB_CHANNEL），
    数值不要求 bit-exact，检查误差在合理范围内。
    """
    print(f"\n{'='*60}")
    print(f"[Forward] num_gemms={num_gemms}, in={in_features}, out={out_features}, "
          f"m_per_expert={m_per_expert}")
    print(f"{'='*60}")

    m_splits = [m_per_expert] * num_gemms
    total_m  = sum(m_splits)
    nvfp4_recipe = make_nvfp4_recipe()

    # 构建模型
    te_model   = build_te_grouped_linear(num_gemms, in_features, out_features, bias, seed)
    mini_model = build_mini_grouped_linear(num_gemms, in_features, out_features, bias, seed, qparam)

    # 整数 weight 初始化（消除 amax 差异）
    init_integer_weights(te_model,   num_gemms, seed + 10)
    init_integer_weights(mini_model, num_gemms, seed + 10)
    # mini 从 TE 同步 weight，保证完全一致
    sync_weights_te_to_mini(te_model, mini_model, num_gemms)

    # 整数 input
    torch.manual_seed(seed + 100)
    inp = torch.randint(-1, 2, (total_m, in_features), device="cuda").to(torch.bfloat16)

    # TE forward
    with torch.no_grad():
        with te.autocast(enabled=True, recipe=nvfp4_recipe):
            out_te = te_model(inp.clone(), m_splits, is_first_microbatch=True)

    # mini forward
    with torch.no_grad():
        out_mini = mini_model(inp.clone(), m_splits=m_splits, is_first_microbatch=True)

    # 对比
    diff     = (out_te - out_mini).abs()
    rel_diff = diff / (out_te.abs() + 1e-6)

    print(f"  TE   output shape:  {out_te.shape}   dtype={out_te.dtype}")
    print(f"  mini output shape:  {out_mini.shape}  dtype={out_mini.dtype}")
    print(f"  max abs diff:       {diff.max():.4e}")
    print(f"  mean abs diff:      {diff.mean():.4e}")
    print(f"  max rel diff:       {rel_diff.max():.4e}")
    print(f"  mean rel diff:      {rel_diff.mean():.4e}")

    torch.testing.assert_close(
        out_te, out_mini,
        rtol=rtol, atol=atol,
        msg=f"[Forward] TE vs mini mismatch"
    )
    print(f"  ✓ Forward passed (atol={atol})")


def test_backward_dgrad(
    num_gemms: int = 4,
    in_features: int = 512,
    out_features: int = 512,
    m_per_expert: int = 64,
    bias: bool = False,
    seed: int = 42,
    qparam: QuantizeRecipe = QuantizeRecipe.FP4_SUB_CHANNEL,
    rtol: float = 0.5,
    atol: float = 0.5,
):
    """
    Backward dgrad 精度对比：miniTransformer vs TE。
    """
    print(f"\n{'='*60}")
    print(f"[Backward dgrad] num_gemms={num_gemms}, in={in_features}, out={out_features}")
    print(f"{'='*60}")

    m_splits = [m_per_expert] * num_gemms
    total_m  = sum(m_splits)
    nvfp4_recipe = make_nvfp4_recipe()

    te_model   = build_te_grouped_linear(num_gemms, in_features, out_features, bias, seed)
    mini_model = build_mini_grouped_linear(num_gemms, in_features, out_features, bias, seed, qparam)

    init_integer_weights(te_model,   num_gemms, seed + 10)
    init_integer_weights(mini_model, num_gemms, seed + 10)
    sync_weights_te_to_mini(te_model, mini_model, num_gemms)

    # 整数 input 和 dy
    torch.manual_seed(seed + 100)
    inp = torch.randint(-1, 2, (total_m, in_features), device="cuda").to(torch.bfloat16)
    torch.manual_seed(seed + 200)
    dy  = torch.randint(-1, 2, (total_m, out_features), device="cuda").to(torch.bfloat16)

    # TE backward
    inp_te = inp.clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=nvfp4_recipe):
        out_te = te_model(inp_te, m_splits, is_first_microbatch=True)
    out_te.backward(dy.clone())
    dgrad_te = inp_te.grad.clone()

    # mini backward
    inp_mini = inp.clone().requires_grad_(True)
    out_mini = mini_model(inp_mini.clone(), m_splits=m_splits, is_first_microbatch=True)
    out_mini.backward(dy.clone())
    dgrad_mini = inp_mini.grad.clone()

    # 对比
    diff     = (dgrad_te - dgrad_mini).abs()
    rel_diff = diff / (dgrad_te.abs() + 1e-6)

    print(f"  TE   dgrad shape:   {dgrad_te.shape}")
    print(f"  mini dgrad shape:   {dgrad_mini.shape}")
    print(f"  max abs diff:       {diff.max():.4e}")
    print(f"  mean abs diff:      {diff.mean():.4e}")
    print(f"  max rel diff:       {rel_diff.max():.4e}")
    print(f"  mean rel diff:      {rel_diff.mean():.4e}")

    torch.testing.assert_close(
        dgrad_te, dgrad_mini,
        rtol=rtol, atol=atol,
        msg=f"[Backward dgrad] TE vs mini mismatch"
    )
    print(f"  ✓ Backward dgrad passed (atol={atol})")


def test_backward_wgrad(
    num_gemms: int = 4,
    in_features: int = 512,
    out_features: int = 512,
    m_per_expert: int = 64,
    bias: bool = False,
    seed: int = 42,
    qparam: QuantizeRecipe = QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET,
    rtol: float = 0.5,
    atol: float = 0.5,
):
    """
    Backward wgrad 精度对比：miniTransformer vs TE。
    """
    print(f"\n{'='*60}")
    print(f"[Backward wgrad] num_gemms={num_gemms}, in={in_features}, out={out_features}")
    print(f"{'='*60}")

    m_splits = [m_per_expert] * num_gemms
    total_m  = sum(m_splits)
    nvfp4_recipe = make_nvfp4_recipe()

    te_model   = build_te_grouped_linear(num_gemms, in_features, out_features, bias, seed)
    mini_model = build_mini_grouped_linear(
        num_gemms, in_features, out_features, bias, seed, qparam,
        fuse_wgrad_accumulation=True,   # wgrad 测试需要 main_grad
    )

    # mini 需要 main_grad 才能做 fuse_wgrad_accumulation
    init_integer_weights(te_model,   num_gemms, seed + 10)
    init_integer_weights(mini_model, num_gemms, seed + 10)
    sync_weights_te_to_mini(te_model, mini_model, num_gemms)

    for i in range(num_gemms):
        w_mini = getattr(mini_model, f"weight{i}")
        w_mini.main_grad = torch.zeros_like(w_mini.data, dtype=torch.float32)

    torch.manual_seed(seed + 100)
    inp = torch.randint(-1, 2, (total_m, in_features), device="cuda").to(torch.bfloat16)
    torch.manual_seed(seed + 200)
    dy  = torch.randint(-1, 2, (total_m, out_features), device="cuda").to(torch.bfloat16)

    # TE backward（weight.grad）
    inp_te = inp.clone().requires_grad_(True)
    with te.autocast(enabled=True, recipe=nvfp4_recipe):
        out_te = te_model(inp_te, m_splits, is_first_microbatch=True)
    out_te.backward(dy.clone())

    # mini backward（weight.main_grad）
    inp_mini = inp.clone().requires_grad_(True)
    out_mini = mini_model(inp_mini.clone(), m_splits=m_splits, is_first_microbatch=True)
    out_mini.backward(dy.clone())

    # 对比每个 expert 的 wgrad
    all_passed = True
    for i in range(num_gemms):
        wgrad_te   = getattr(te_model, f"weight{i}").grad
        wgrad_mini = getattr(mini_model, f"weight{i}").main_grad

        if wgrad_te is None:
            print(f"  expert {i}: TE wgrad is None, skipping")
            continue
        if wgrad_mini is None:
            print(f"  expert {i}: mini wgrad is None, skipping")
            continue

        # dtype 对齐（TE grad 可能是 bf16，mini main_grad 是 fp32）
        wgrad_te_f = wgrad_te.float()
        wgrad_mini_f = wgrad_mini.float()

        diff     = (wgrad_te_f - wgrad_mini_f).abs()
        rel_diff = diff / (wgrad_te_f.abs() + 1e-6)

        print(f"  expert {i}: wgrad shape={wgrad_mini.shape}  "
              f"max={diff.max():.4e}  mean={diff.mean():.4e}  "
              f"max_rel={rel_diff.max():.4e}")
        print(f"    TE   wgrad: min={wgrad_te_f.min():.4f} max={wgrad_te_f.max():.4f} "
              f"mean={wgrad_te_f.mean():.4f} dtype={wgrad_te.dtype}")
        print(f"    mini wgrad: min={wgrad_mini_f.min():.4f} max={wgrad_mini_f.max():.4f} "
              f"mean={wgrad_mini_f.mean():.4f} dtype={wgrad_mini.dtype}")

        try:
            torch.testing.assert_close(
                wgrad_te_f, wgrad_mini_f,
                rtol=rtol, atol=atol,
                msg=f"[Backward wgrad] expert {i} TE vs mini mismatch"
            )
        except AssertionError as e:
            print(f"  ✗ expert {i} failed: {e}")
            all_passed = False

    if all_passed:
        print(f"  ✓ Backward wgrad passed (atol={atol})")
    else:
        raise AssertionError("[Backward wgrad] some experts failed, see above")


# ==================== Main ====================

ALL_TESTS = [
    ("Forward",          test_forward),
    ("Backward dgrad",   test_backward_dgrad),
    ("Backward wgrad",   test_backward_wgrad),
]


def run_all_tests(
    num_gemms: int = 4,
    in_features: int = 512,
    out_features: int = 512,
    m_per_expert: int = 64,
    qparam: QuantizeRecipe = QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET,
    rtol: float = 0.5,
    atol: float = 0.5,
):
    print("\n" + "=" * 60)
    print("GroupedLinear NVFP4 精度对比：miniTransformer vs TransformerEngine")
    print("=" * 60)
    print(f"  num_gemms:    {num_gemms}")
    print(f"  in_features:  {in_features}")
    print(f"  out_features: {out_features}")
    print(f"  m_per_expert: {m_per_expert}")
    print(f"  qparam:       {qparam}")
    print(f"  atol:         {atol}")
    print("=" * 60)

    kwargs = dict(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        m_per_expert=m_per_expert,
        qparam=qparam,
        rtol=rtol,
        atol=atol,
    )

    for i, (name, fn) in enumerate(ALL_TESTS, 1):
        print(f"\n[{i}/{len(ALL_TESTS)}] {name}")
        try:
            fn(**kwargs)
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            traceback.print_exc()
            raise

    print("\n" + "=" * 60)
    print(f"✓ 全部 {len(ALL_TESTS)} 项测试通过！")
    print("=" * 60 + "\n")


QPARAM_MAP = {
    "non":      QuantizeRecipe.NON_QUANTIZE,
    "int4":     QuantizeRecipe.INT4_SUB_CHANNEL,
    "fp4_ref":  QuantizeRecipe.FP4_SUB_CHANNEL_REF,
    "fp4":      QuantizeRecipe.FP4_SUB_CHANNEL,
    "fp4_q":    QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GroupedLinear NVFP4 精度对比测试：miniTransformer vs TransformerEngine"
    )
    parser.add_argument("--num-gemms",    type=int,   default=4)
    parser.add_argument("--in-features",  type=int,   default=512)
    parser.add_argument("--out-features", type=int,   default=512)
    parser.add_argument("--m-per-expert", type=int,   default=64)
    parser.add_argument("--qparam",       type=str,   default="fp4",
                        choices=list(QPARAM_MAP.keys()),
                        help="mini 侧量化 recipe（默认 fp4）")
    parser.add_argument("--atol",         type=float, default=0.5)
    parser.add_argument("--rtol",         type=float, default=0.5)
    args = parser.parse_args()

    run_all_tests(
        num_gemms=args.num_gemms,
        in_features=args.in_features,
        out_features=args.out_features,
        m_per_expert=args.m_per_expert,
        qparam=QPARAM_MAP[args.qparam],
        rtol=args.rtol,
        atol=args.atol,
    )
