"""GroupedLinear smoke tests across quantization recipes (forward / backward / splits).

Usage:
    python test_group_linear_exact.py
    python test_group_linear_exact.py --tests 1 2 3
    python test_group_linear_exact.py --recipe nvfp4_per_token
    python test_group_linear_exact.py --qparam 6  # legacy test ID, not QAT6
    python test_group_linear_exact.py --tests 1 4 --qparam 9
"""
import argparse
import traceback

import torch
from torch.profiler import profile, ProfilerActivity, record_function

from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.group_linear import GroupedLinear
from miniTransformer.recipes import get_recipe

AVAILABLE_TESTS = {
    1: ("test_group_linear_forward", "Forward pass execution"),
    2: ("test_group_linear_backward", "Backward pass with gradients"),
    3: ("test_group_linear_varying_splits", "Varying m_splits configurations"),
    4: ("test_group_linear_multiple_microbatches", "Multiple microbatches (weight caching)"),
    5: ("test_group_linear_perf", "Performance profiling with large inputs"),
    6: ("test_group_linear_logprob_vs_train", "Forward-only (no_grad) vs train: w-transpose skip"),
    7: ("test_group_linear_input_row_amax", "GroupedLinear forward/backward with prefused row_amax"),
}

# Historical test IDs are independent of runtime QAT_PARAMS; preserve them.
AVAILABLE_QPARAMS = {
    1: ("NON_QUANTIZE", QuantizeRecipe.NON_QUANTIZE, "No quantization"),
    2: ("INT4_SUB_CHANNEL", QuantizeRecipe.INT4_SUB_CHANNEL, "INT4 sub-channel quantization"),
    3: ("FP4_SUB_CHANNEL_REF", QuantizeRecipe.FP4_SUB_CHANNEL_REF, "FP4 reference implementation"),
    4: ("FP4_SUB_CHANNEL", QuantizeRecipe.FP4_SUB_CHANNEL, "FP4 tiled quantization"),
    5: ("FP4_SUB_CHANNEL_QUARTET", QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET, "FP4 quartet quantization"),
    6: ("FP4_SUB_CHANNEL_CURSOR2", QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2, "FP4 cursor2 recipe"),
    7: ("FP4_SUB_CHANNEL_CURSOR2_CUTLASS", QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2_CUTLASS, "FP4 cursor2 CUTLASS kernel"),
    8: ("FP4_SUB_CHANNEL_CURSOR2_CUTLASS_BWD", QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2_CUTLASS_BWD, "FP4 cursor2 CUTLASS + MXFP8 backward"),
    9: ("FP4_SUB_CHANNEL_PERTOKEN", QuantizeRecipe.FP4_SUB_CHANNEL_PERTOKEN, "Per-token NVFP4 amax-fuse path"),
}


class TestGroupLinearExact:
    """GroupedLinear smoke suite."""

    def build_group_linear(
        self,
        num_gemms: int = 4,
        in_features: int = 4096,
        out_features: int = 4096,
        qparam: QuantizeRecipe = None,
    ):
        qlinear_params = get_qlinear_params_from_predefined(qparam)
        return GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            sequence_parallel=False,
            fuse_wgrad_accumulation=True,
            tp_group=None,
            tp_size=1,
            get_rng_state_tracker=None,
            rng_tracker_name=None,
            init_method=None,
            bias=False,
            return_bias=False,
            params_dtype=torch.bfloat16,
            parallel_mode=None,
            device="cuda",
            ub_overlap_rs=False,
            ub_overlap_ag=False,
            ub_name="test_group_linear",
            layer_number=0,
            qlinear_params=qlinear_params,
        )

    def _init_main_grads(self, group_linear: GroupedLinear, num_gemms: int) -> None:
        for i in range(num_gemms):
            weight = getattr(group_linear, f"weight{i}")
            weight.main_grad = torch.zeros_like(weight.data, dtype=torch.float32)

    def test_group_linear_forward(self, qparam: QuantizeRecipe):
        print(f"\n{'='*60}")
        print("Testing GroupLinear Forward Pass")
        print(f"Quantization Recipe: {qparam}")
        print(f"{'='*60}")

        num_gemms = 4
        in_features = 4096
        out_features = 4096
        batch_size = 1024
        m_splits = [256, 256, 256, 256]
        assert sum(m_splits) == batch_size

        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam,
        )
        inp = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)

        print(f"Input shape: {inp.shape}")
        print(f"m_splits: {m_splits}")
        print(f"Number of GEMMs: {num_gemms}")

        out = group_linear(inp, m_splits=m_splits, is_first_microbatch=True)
        print(f"Output shape: {out.shape}")
        print("GroupLinear forward pass successful!")
        print(f"{'='*60}\n")
        return group_linear, inp, out

    def test_group_linear_backward(self, qparam: QuantizeRecipe):
        print(f"\n{'='*60}")
        print("Testing GroupLinear Backward Pass")
        print(f"Quantization Recipe: {qparam}")
        print(f"{'='*60}")

        num_gemms = 4
        in_features = 4096
        out_features = 4096
        batch_size = 1024
        m_splits = [256, 256, 256, 256]

        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam,
        )
        self._init_main_grads(group_linear, num_gemms)

        inp = torch.randn(
            batch_size, in_features, device="cuda", dtype=torch.bfloat16
        ).requires_grad_(True)

        print(f"Input shape: {inp.shape}")
        print(f"m_splits: {m_splits}")
        print(f"Number of GEMMs: {num_gemms}")

        out = group_linear(inp, m_splits=m_splits, is_first_microbatch=True)
        print(f"Output shape: {out.shape}")
        out.sum().backward()

        assert inp.grad is not None, "Input gradient should not be None"
        for i in range(num_gemms):
            weight = getattr(group_linear, f"weight{i}")
            assert weight.main_grad is not None, f"Weight{i} main_grad should not be None"

        print(f"Input gradient shape: {inp.grad.shape}")
        print("GroupLinear backward pass successful!")
        print(f"{'='*60}\n")

    def test_group_linear_varying_splits(self, qparam: QuantizeRecipe):
        print(f"\n{'='*60}")
        print("Testing GroupLinear with Varying Splits")
        print(f"Quantization Recipe: {qparam}")
        print(f"{'='*60}")

        num_gemms = 3
        in_features = 2048
        out_features = 2048
        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam,
        )

        test_configs = [
            ([512, 512, 512], "Equal splits"),
            ([256, 768, 512], "Unequal splits"),
            ([1024, 256, 256], "First large"),
        ]
        for m_splits, desc in test_configs:
            batch_size = sum(m_splits)
            print(f"\nTesting: {desc}")
            print(f"  m_splits: {m_splits}, total: {batch_size}")
            inp = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
            out = group_linear(inp, m_splits=m_splits, is_first_microbatch=True)
            print(f"  Input shape: {inp.shape}, Output shape: {out.shape}")
            assert out.shape[0] == batch_size, "Output batch size mismatch"
            print("  Passed!")

        print("\nAll split configurations successful!")
        print(f"{'='*60}\n")

    def test_group_linear_multiple_microbatches(self, qparam: QuantizeRecipe):
        print(f"\n{'='*60}")
        print("Testing GroupLinear Multiple Microbatches")
        print(f"Quantization Recipe: {qparam}")
        print(f"{'='*60}")

        num_gemms = 2
        in_features = 2048
        out_features = 2048
        batch_size = 512
        m_splits = [256, 256]
        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam,
        )

        print("\nMicrobatch 1 (is_first_microbatch=True):")
        out1 = group_linear(
            torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16),
            m_splits=m_splits,
            is_first_microbatch=True,
        )
        print(f"  Output shape: {out1.shape}")

        print("\nMicrobatch 2 (is_first_microbatch=False):")
        out2 = group_linear(
            torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16),
            m_splits=m_splits,
            is_first_microbatch=False,
        )
        print(f"  Output shape: {out2.shape}")

        print("\nMicrobatch 3 (is_first_microbatch=None):")
        out3 = group_linear(
            torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16),
            m_splits=m_splits,
            is_first_microbatch=None,
        )
        print(f"  Output shape: {out3.shape}")

        print("\nMultiple microbatches test successful!")
        print(f"{'='*60}\n")

    def test_group_linear_perf(
        self,
        qparam: QuantizeRecipe,
        num_gemms: int = 64,
        in_features: int = 4096,
        out_features: int = 4096,
        batch_size: int = 4096,
        warmup_iters: int = 3,
        profile_iters: int = 5,
        trace_path: str = "/tmp/gl_perf_trace.json",
        nsys_only: bool = False,
    ):
        """Warmup then profile large GroupedLinear steps; optional nsys-only CUPTI path."""
        print(f"\n{'='*60}")
        print("Testing GroupLinear Performance")
        print(f"Quantization Recipe: {qparam}")
        print(f"num_gemms={num_gemms}  in={in_features}  out={out_features}  batch={batch_size}")
        print(f"warmup={warmup_iters}  profile_iters={profile_iters}  nsys_only={nsys_only}")
        print(f"{'='*60}")

        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam,
        )
        self._init_main_grads(group_linear, num_gemms)

        tokens_per_expert = batch_size // num_gemms
        m_splits = [tokens_per_expert] * num_gemms

        def make_inp():
            return torch.randn(
                batch_size, in_features, device="cuda", dtype=torch.bfloat16
            ).requires_grad_(True)

        def step(inp):
            out = group_linear(inp, m_splits=m_splits, is_first_microbatch=True)
            out.sum().backward()

        print(f"\nWarmup ({warmup_iters} iters)...")
        for _ in range(warmup_iters):
            step(make_inp())
        torch.cuda.synchronize()
        print("Warmup done.")

        print(f"\nProfiling ({profile_iters} iters)...")
        torch.cuda.cudart().cudaProfilerStart()

        if nsys_only:
            for _ in range(profile_iters):
                with record_function("full_fwd_bwd"):
                    step(make_inp())
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print("[Perf] nsys-only mode: torch.profiler skipped.")
            print(f"{'='*60}\n")
            return

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in range(profile_iters):
                with record_function("full_fwd_bwd"):
                    step(make_inp())

        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()

        avgs = prof.key_averages()
        print(f"\n{'='*60}")
        print("key_averages — sorted by CPU total (top 80):")
        print(f"{'='*60}")
        print(avgs.table(sort_by="cpu_time_total", row_limit=80))

        print(f"\n{'='*60}")
        print("key_averages — sorted by Self CPU (top 40):")
        print(f"{'='*60}")
        print(avgs.table(sort_by="self_cpu_time_total", row_limit=40))

        prof.export_chrome_trace(trace_path)
        print(f"[Perf] Chrome trace -> {trace_path}")
        print(f"{'='*60}\n")

    def test_group_linear_logprob_vs_train(
        self,
        qparam: QuantizeRecipe,
        num_gemms: int = 64,
        in_features: int = 4096,
        out_features: int = 4096,
        batch_size: int = 4096,
        warmup_iters: int = 3,
        profile_iters: int = 10,
    ):
        """Compare no_grad forward (log_prob style) vs train forward+backward CUDA cost."""
        print(f"\n{'='*60}")
        print("Testing GroupLinear log_prob (no_grad) vs train forward")
        print(f"Quantization Recipe: {qparam}")
        print(f"num_gemms={num_gemms}  in={in_features}  out={out_features}  batch={batch_size}")
        print(f"{'='*60}")

        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam,
        )
        self._init_main_grads(group_linear, num_gemms)

        tokens_per_expert = batch_size // num_gemms
        m_splits = [tokens_per_expert] * num_gemms

        def make_inp(requires_grad):
            return torch.randn(
                batch_size, in_features, device="cuda", dtype=torch.bfloat16
            ).requires_grad_(requires_grad)

        def run_forward_only():
            group_linear.eval()
            with torch.no_grad():
                return group_linear(
                    make_inp(False), m_splits=m_splits, is_first_microbatch=None
                )

        def run_train():
            group_linear.train()
            out = group_linear(make_inp(True), m_splits=m_splits, is_first_microbatch=True)
            out.sum().backward()
            return out

        def measure(label, fn):
            for _ in range(warmup_iters):
                fn()
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                for _ in range(profile_iters):
                    fn()
                torch.cuda.synchronize()

            events = prof.events()
            qw_cuda_us = 0.0
            kernel_us = {}
            for ev in events:
                name = ev.key
                dev = getattr(ev, "cuda_time_total", 0) or 0
                if any(
                    k in name
                    for k in (
                        "quantize",
                        "fp4_group",
                        "transpose",
                        "swizzle",
                        "multi_padding",
                        "CatArrayBatched",
                        "reduce_kernel",
                    )
                ):
                    if dev > 0:
                        kernel_us[name] = kernel_us.get(name, 0.0) + dev
                        qw_cuda_us += dev
            print(f"\n--- {label} ---")
            print(
                f"  quant-related CUDA total: {qw_cuda_us/1e3:.3f} ms "
                f"({qw_cuda_us/1e3/profile_iters:.3f} ms/iter)"
            )
            top = sorted(kernel_us.items(), key=lambda x: -x[1])[:12]
            for kname, us in top:
                short = kname if len(kname) <= 70 else kname[:67] + "..."
                print(f"    {us/1e3:8.3f} ms  {short}")
            return qw_cuda_us

        train_us = measure("TRAIN forward+backward (is_grad_enabled=True)", run_train)
        lp_us = measure("LOG_PROB forward-only (no_grad, is_grad_enabled=False)", run_forward_only)

        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"  train     quant CUDA: {train_us/1e3:.3f} ms")
        print(f"  log_prob  quant CUDA: {lp_us/1e3:.3f} ms")
        if train_us > 0:
            print(
                f"  log_prob / train ratio: {lp_us/train_us:.3f}  "
                f"(lower = more w-transpose quant skipped)"
            )
        print(f"{'='*60}\n")

    def test_group_linear_input_row_amax(self, qparam: QuantizeRecipe):
        """Smoke: GroupedLinear with prefused input_row_amax (rowwise skip K1)."""
        if qparam != QuantizeRecipe.FP4_SUB_CHANNEL_PERTOKEN:
            print(
                f"Skipping row_amax test for recipe {qparam} "
                "(requires FP4_SUB_CHANNEL_PERTOKEN)"
            )
            return

        print(f"\n{'='*60}")
        print("Testing GroupedLinear with input_row_amax (amax-fuse path)")
        print(f"Quantization Recipe: {qparam}")
        print(f"{'='*60}")

        num_gemms = 4
        in_features = 4096
        out_features = 4096
        batch_size = 1024
        m_splits = [256, 256, 256, 256]

        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam,
        )
        self._init_main_grads(group_linear, num_gemms)

        torch.manual_seed(1234)
        inp = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
        row_amax = inp.detach().abs().amax(dim=-1).to(torch.float32)

        out_baseline = group_linear(inp, m_splits=m_splits, is_first_microbatch=True)
        out_fused = group_linear(
            inp,
            m_splits=m_splits,
            is_first_microbatch=True,
            input_row_amax=row_amax,
        )

        print(f"Output shape: {out_fused.shape}")
        max_fwd_diff = (out_baseline.float() - out_fused.float()).abs().max().item()
        print(f"Forward max |baseline - fused|: {max_fwd_diff:.6g}")
        assert torch.isfinite(out_fused).all(), "Fused forward output contains non-finite values"
        torch.testing.assert_close(out_baseline, out_fused, rtol=0, atol=0)

        inp_grad = inp.clone().requires_grad_(True)
        out = group_linear(
            inp_grad,
            m_splits=m_splits,
            is_first_microbatch=True,
            input_row_amax=row_amax,
        )
        out.sum().backward()

        assert inp_grad.grad is not None, "Input gradient should not be None"
        for i in range(num_gemms):
            weight = getattr(group_linear, f"weight{i}")
            assert weight.main_grad is not None, f"Weight{i} main_grad should not be None"

        print("Input gradient shape:", inp_grad.grad.shape)
        print("GroupedLinear input_row_amax forward/backward successful!")
        print(f"{'='*60}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="GroupLinear Module Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Tests:
  1 - Forward pass execution
  2 - Backward pass with gradients
  3 - Varying m_splits configurations
  4 - Multiple microbatches (weight caching)
  5 - Performance profiling
  6 - log_prob vs train CUDA compare
  7 - input_row_amax (PERTOKEN only)

Legacy test recipe IDs (independent of QAT_PARAMS):
  1 - NON_QUANTIZE
  2 - INT4_SUB_CHANNEL
  3 - FP4_SUB_CHANNEL_REF
  4 - FP4_SUB_CHANNEL (default)
  5 - FP4_SUB_CHANNEL_QUARTET
  6 - FP4_SUB_CHANNEL_CURSOR2
  7 - FP4_SUB_CHANNEL_CURSOR2_CUTLASS
  8 - FP4_SUB_CHANNEL_CURSOR2_CUTLASS_BWD
  9 - FP4_SUB_CHANNEL_PERTOKEN

Examples:
  python test_group_linear_exact.py
  python test_group_linear_exact.py --tests 1 2 3
  python test_group_linear_exact.py --qparam 6
  python test_group_linear_exact.py --tests 1 7 --qparam 9
        """,
    )
    parser.add_argument(
        "--tests",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5, 6, 7],
        default=None,
        help="Test IDs to run (default: 1-4 correctness tests)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--recipe", help="Public recipe name or enum value (recommended)")
    selection.add_argument(
        "--qparam",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        default=None,
        help="Legacy test recipe ID, NOT a QAT_PARAMS ID (see --list-qparams). Default: 4",
    )
    parser.add_argument("--list-tests", action="store_true", help="List tests and exit")
    parser.add_argument("--list-qparams", action="store_true", help="List recipes and exit")
    parser.add_argument(
        "--nsys-only",
        action="store_true",
        help="Skip torch.profiler in perf test for exclusive CUPTI access",
    )
    return parser.parse_args()


def list_available_tests():
    print("\nAvailable Tests:")
    print("=" * 60)
    for test_id, (test_name, test_desc) in AVAILABLE_TESTS.items():
        print(f"  {test_id}. {test_desc}")
        print(f"     Function: {test_name}")
    print("=" * 60)


def list_available_qparams():
    print("\nLegacy test recipe IDs (independent of QAT_PARAMS):")
    print("=" * 60)
    for qparam_id, (qparam_name, _, qparam_desc) in AVAILABLE_QPARAMS.items():
        print(f"  {qparam_id}. {qparam_name}")
        print(f"     {qparam_desc}")
    print("=" * 60)


def run_all_tests(qparam_id: int = 4, test_ids: list = None, nsys_only: bool = False, recipe_name: str = None):
    print("\n" + "=" * 60)
    print("GroupLinear Module Test Suite")
    print("=" * 60)

    if recipe_name is not None:
        spec = get_recipe(recipe_name)
        qparam_name = spec.name
        qparam_recipe = QuantizeRecipe(spec.enum_value)
        qparam_desc = spec.description
        recipe_label = f"{spec.name} (QAT{spec.qat_id})" if spec.qat_id is not None else spec.name
    else:
        qparam_name, qparam_recipe, qparam_desc = AVAILABLE_QPARAMS[qparam_id]
        recipe_label = f"legacy test ID {qparam_id}: {qparam_name} (not QAT_PARAMS)"
    print(f"Quantization Recipe: {recipe_label}")
    print(f"Description: {qparam_desc}")

    if test_ids is None:
        test_ids = [1, 2, 3, 4]
    else:
        test_ids = sorted(test_ids)

    print(f"Tests to run: {test_ids}")
    print("=" * 60)

    test = TestGroupLinearExact()
    passed_tests = []
    failed_tests = []

    for test_id in test_ids:
        test_name, test_desc = AVAILABLE_TESTS[test_id]
        try:
            print(f"\n>>> Running Test {test_id}: {test_desc}")
            test_method = getattr(test, test_name)
            if test_id == 5:
                test_method(qparam_recipe, nsys_only=nsys_only)
            else:
                test_method(qparam_recipe)
            passed_tests.append((test_id, test_desc))
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"Test {test_id} FAILED: {test_desc}")
            print(f"{'='*60}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print("\nFull Traceback:")
            print("-" * 60)
            traceback.print_exc()
            print("-" * 60)
            failed_tests.append((test_id, test_desc, e))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Quantization: {recipe_label}")
    print(f"Total tests run: {len(test_ids)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")

    if passed_tests:
        print("\nPassed tests:")
        for test_id, test_desc in passed_tests:
            print(f"  {test_id}. {test_desc}")

    if failed_tests:
        print("\nFailed tests:")
        for test_id, test_desc, error in failed_tests:
            print(f"  {test_id}. {test_desc}")
            print(f"     Error: {type(error).__name__}: {error}")

    print("=" * 60 + "\n")

    if failed_tests:
        print(f"{len(failed_tests)} test(s) failed. See detailed traceback above.\n")
        exit(1)
    print("All tests passed successfully!\n")


if __name__ == "__main__":
    args = parse_arguments()
    if args.list_tests:
        list_available_tests()
        exit(0)
    if args.list_qparams:
        list_available_qparams()
        exit(0)
    run_all_tests(qparam_id=args.qparam if args.qparam is not None else 4,
                  test_ids=args.tests, nsys_only=args.nsys_only, recipe_name=args.recipe)
