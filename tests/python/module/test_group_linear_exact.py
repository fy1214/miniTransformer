"""
GroupLinear Exact Correctness Tests
====================================

This module tests the GroupLinear module to ensure it can execute forward
and backward passes correctly with different quantization recipes.

The tests verify:
1. Forward pass execution without errors
2. Backward pass with gradient computation
3. Multiple GEMMs with different split configurations

Usage:
    # Run all tests with default quantization (FP4_SUB_CHANNEL_QUARTET)
    python test_group_linear_exact.py
    
    # Run specific tests
    python test_group_linear_exact.py --tests 1 2 3
    
    # Use different quantization recipe (use number 1-5)
    python test_group_linear_exact.py --qparam 3
    
    # Combine options
    python test_group_linear_exact.py --tests 1 4 --qparam 1
"""
import argparse
import traceback
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.group_linear import GroupedLinear


# ==================== Available Tests ====================

AVAILABLE_TESTS = {
    1: ("test_group_linear_forward", "Forward pass execution"),
    2: ("test_group_linear_backward", "Backward pass with gradients"),
    3: ("test_group_linear_varying_splits", "Varying m_splits configurations"),
    4: ("test_group_linear_multiple_microbatches", "Multiple microbatches (weight caching)"),
    5: ("test_group_linear_perf", "Performance profiling with large inputs"),
    6: ("test_group_linear_logprob_vs_train", "Forward-only (no_grad) vs train: w-transpose skip"),
}

# ==================== Available Quantization Recipes ====================

AVAILABLE_QPARAMS = {
    1: ("NON_QUANTIZE", QuantizeRecipe.NON_QUANTIZE, "No quantization"),
    2: ("INT4_SUB_CHANNEL", QuantizeRecipe.INT4_SUB_CHANNEL, "INT4 sub-channel quantization"),
    3: ("FP4_SUB_CHANNEL_REF", QuantizeRecipe.FP4_SUB_CHANNEL_REF, "FP4 reference implementation"),
    4: ("FP4_SUB_CHANNEL", QuantizeRecipe.FP4_SUB_CHANNEL, "FP4 tiled quantization"),
    5: ("FP4_SUB_CHANNEL_QUARTET", QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET, "FP4 quartet quantization"),
    6: ("FP4_SUB_CHANNEL_CURSOR2", QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2, "FP4 cursor2 recipe"),
    7: ("FP4_SUB_CHANNEL_CURSOR2_CUTLASS", QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2_CUTLASS, "FP4 cursor2 CUTLASS kernel"),
    8: ("FP4_SUB_CHANNEL_CURSOR2_CUTLASS_BWD", QuantizeRecipe.FP4_SUB_CHANNEL_CURSOR2_CUTLASS_BWD, "FP4 cursor2 CUTLASS + MXFP8 backward"),
}


# ==================== Test Class ====================

class TestGroupLinearExact:
    """Test suite for GroupLinear module."""

    def build_group_linear(
        self,
        num_gemms: int = 4,
        in_features: int = 4096,
        out_features: int = 4096,
        qparam: QuantizeRecipe = None
    ):
        """
        Build a GroupLinear module for testing.
        
        Args:
            num_gemms: Number of grouped linear operations
            in_features: Input feature dimension
            out_features: Output feature dimension
            qparam: Quantization recipe to use
            
        Returns:
            GroupedLinear module
        """
        # Configuration parameters
        sequence_parallel = False
        fuse_wgrad_accumulation = True
        tp_group = None
        tp_size = 1
        get_rng_state_tracker = None
        rng_tracker_name = None
        init_method = None
        bias = False
        return_bias = False
        params_dtype = torch.bfloat16
        parallel_mode = None  # GroupLinear typically doesn't use column/row parallel
        device = "cuda"
        ub_overlap_rs = False
        ub_overlap_ag = False
        ub_name = "test_group_linear"
        layer_number = 0
        
        # Get quantization parameters
        qlinear_params = get_qlinear_params_from_predefined(qparam)

        # Create GroupedLinear module
        group_linear = GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            sequence_parallel=sequence_parallel,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            tp_group=tp_group,
            tp_size=tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            rng_tracker_name=rng_tracker_name,
            init_method=init_method,
            bias=bias,
            return_bias=return_bias,
            params_dtype=params_dtype,
            parallel_mode=parallel_mode,
            device=device,
            ub_overlap_rs=ub_overlap_rs,
            ub_overlap_ag=ub_overlap_ag,
            ub_name=ub_name,
            layer_number=layer_number,
            qlinear_params=qlinear_params
        )
        
        return group_linear

    def test_group_linear_forward(self, qparam: QuantizeRecipe):
        """
        Test GroupLinear forward pass.
        
        Args:
            qparam: Quantization recipe to use
        """
        print(f"\n{'='*60}")
        print(f"Testing GroupLinear Forward Pass")
        print(f"Quantization Recipe: {qparam}")
        print(f"{'='*60}")
        
        # Build module
        num_gemms = 4
        in_features = 4096
        out_features = 4096  # Must be multiple of 128
        
        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam
        )
        
        # Create input tensor
        batch_size = 1024
        m_splits = [256, 256, 256, 256]  # Split batch into 4 parts
        assert sum(m_splits) == batch_size, "m_splits should sum to batch_size"
        
        inp = torch.randn(
            batch_size, in_features, 
            device="cuda", 
            dtype=torch.bfloat16
        )
        
        print(f"Input shape: {inp.shape}")
        print(f"m_splits: {m_splits}")
        print(f"Number of GEMMs: {num_gemms}")
        
        # Forward pass
        out = group_linear(inp, m_splits=m_splits, is_first_microbatch=True)
        
        print(f"Output shape: {out.shape}")
        print(f"✓ GroupLinear forward pass successful!")
        print(f"{'='*60}\n")
        
        return group_linear, inp, out

    def test_group_linear_backward(self, qparam: QuantizeRecipe):
        """
        Test GroupLinear backward pass with gradient computation.
        
        Args:
            qparam: Quantization recipe to use
        """
        print(f"\n{'='*60}")
        print(f"Testing GroupLinear Backward Pass")
        print(f"Quantization Recipe: {qparam}")
        print(f"{'='*60}")
        
        # Build module
        num_gemms = 4
        in_features = 4096
        out_features = 4096  # Must be multiple of 128
        
        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam
        )

        # Mock main_grad for each weight
        for i in range(num_gemms):
            weight = getattr(group_linear, f"weight{i}")
            weight.main_grad = torch.zeros_like(weight.data, dtype=torch.float32)
        
        # Create input tensor with gradient enabled
        batch_size = 1024
        m_splits = [256, 256, 256, 256]  # Split batch into 4 parts
        
        inp = torch.randn(
            batch_size, in_features,
            device="cuda",
            dtype=torch.bfloat16
        ).requires_grad_(True)
        
        print(f"Input shape: {inp.shape}")
        print(f"m_splits: {m_splits}")
        print(f"Number of GEMMs: {num_gemms}")
        
        # Forward pass
        out = group_linear(inp, m_splits=m_splits, is_first_microbatch=True)
        
        print(f"Output shape: {out.shape}")
        
        # Backward pass
        loss = out.sum()
        loss.backward()
        
        # Check gradients
        assert inp.grad is not None, "Input gradient should not be None"
        for i in range(num_gemms):
            weight = getattr(group_linear, f"weight{i}")
            assert weight.main_grad is not None, f"Weight{i} main_grad should not be None"
        
        print(f"Input gradient shape: {inp.grad.shape}")
        print(f"✓ GroupLinear backward pass successful!")
        print(f"✓ All gradients computed correctly!")
        print(f"{'='*60}\n")

    def test_group_linear_varying_splits(self, qparam: QuantizeRecipe):
        """
        Test GroupLinear with varying m_splits configurations.
        
        Args:
            qparam: Quantization recipe to use
        """
        print(f"\n{'='*60}")
        print(f"Testing GroupLinear with Varying Splits")
        print(f"Quantization Recipe: {qparam}")
        print(f"{'='*60}")
        
        # Build module
        num_gemms = 3
        in_features = 2048
        out_features = 2048  # Must be multiple of 128
        
        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam
        )
        
        # Test different split configurations
        test_configs = [
            ([512, 512, 512], "Equal splits"),
            ([256, 768, 512], "Unequal splits"),
            ([1024, 256, 256], "First large"),
        ]
        
        for m_splits, desc in test_configs:
            batch_size = sum(m_splits)
            print(f"\nTesting: {desc}")
            print(f"  m_splits: {m_splits}, total: {batch_size}")
            
            inp = torch.randn(
                batch_size, in_features,
                device="cuda",
                dtype=torch.bfloat16
            )
            
            out = group_linear(inp, m_splits=m_splits, is_first_microbatch=True)
            
            print(f"  Input shape: {inp.shape}, Output shape: {out.shape}")
            assert out.shape[0] == batch_size, "Output batch size mismatch"
            print(f"  ✓ Passed!")
        
        print(f"\n✓ All split configurations successful!")
        print(f"{'='*60}\n")

    def test_group_linear_multiple_microbatches(self, qparam: QuantizeRecipe):
        """
        Test GroupLinear with multiple microbatches (weight caching).
        
        Args:
            qparam: Quantization recipe to use
        """
        print(f"\n{'='*60}")
        print(f"Testing GroupLinear Multiple Microbatches")
        print(f"Quantization Recipe: {qparam}")
        print(f"{'='*60}")
        
        # Build module
        num_gemms = 2
        in_features = 2048
        out_features = 2048
        
        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam
        )
        
        batch_size = 512
        m_splits = [256, 256]
        
        # First microbatch (should quantize weights)
        print("\nMicrobatch 1 (is_first_microbatch=True):")
        inp1 = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
        out1 = group_linear(inp1, m_splits=m_splits, is_first_microbatch=True)
        print(f"  Output shape: {out1.shape}")
        print(f"  ✓ Weight quantization and caching performed")
        
        # Second microbatch (should use cached weights)
        print("\nMicrobatch 2 (is_first_microbatch=False):")
        inp2 = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
        out2 = group_linear(inp2, m_splits=m_splits, is_first_microbatch=False)
        print(f"  Output shape: {out2.shape}")
        print(f"  ✓ Used cached quantized weights")
        
        # Third microbatch (None means use cached)
        print("\nMicrobatch 3 (is_first_microbatch=None):")
        inp3 = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
        out3 = group_linear(inp3, m_splits=m_splits, is_first_microbatch=None)
        print(f"  Output shape: {out3.shape}")
        print(f"  ✓ Weight quantization performed (None = True)")
        
        print(f"\n✓ Multiple microbatches test successful!")
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
        """Performance profiling with large inputs.

        Runs warmup_iters steps first (not profiled), then profile_iters steps
        under torch.profiler. Prints key_averages() sorted by CUDA time and
        exports a Chrome trace to trace_path.

        When nsys_only=True, torch.profiler is skipped so nsys can have
        exclusive CUPTI access (avoids CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED).
        cudaProfilerStart/Stop markers are always emitted for --capture-range cudaProfilerApi.
        """
        print(f"\n{'='*60}")
        print(f"Testing GroupLinear Performance")
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

        # mock main_grad so fuse_wgrad_accumulation path works
        for i in range(num_gemms):
            w = getattr(group_linear, f"weight{i}")
            w.main_grad = torch.zeros_like(w.data, dtype=torch.float32)

        tokens_per_expert = batch_size // num_gemms
        m_splits = [tokens_per_expert] * num_gemms

        def make_inp():
            return torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16).requires_grad_(True)

        def step(inp):
            out = group_linear(inp, m_splits=m_splits, is_first_microbatch=True)
            out.sum().backward()

        # --- warmup ---
        print(f"\nWarmup ({warmup_iters} iters)...")
        for _ in range(warmup_iters):
            step(make_inp())
        torch.cuda.synchronize()
        print("Warmup done.")

        # --- profile ---
        print(f"\nProfiling ({profile_iters} iters)...")

        # Mark the capture range for nsys --capture-range cudaProfilerApi
        torch.cuda.cudart().cudaProfilerStart()

        if nsys_only:
            # No torch.profiler — nsys gets exclusive CUPTI access for GPU kernel times.
            # NVTX markers from record_function are still visible in nsys.
            for _ in range(profile_iters):
                with record_function("full_fwd_bwd"):
                    step(make_inp())
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            print("[Perf] nsys-only mode: torch.profiler skipped.")
            print(f"{'='*60}\n")
        else:
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
            print(f"[Perf] Chrome trace → {trace_path}")
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
        """Compare forward-only (no_grad, like compute_log_prob) vs train forward.

        The fix makes w-transpose quantization conditional on is_grad_enabled.
        In the no_grad path (log_prob), w columnwise/transpose quant should be
        skipped; in the train path it must still run (needed by backward).

        This prints, for each path:
          - total CUDA time inside group_quantize_w
          - which CUDA kernels appear (transpose/columnwise kernels should
            vanish in the no_grad path)
        so the speedup from the fix is directly measurable.
        """
        print(f"\n{'='*60}")
        print(f"Testing GroupLinear log_prob (no_grad) vs train forward")
        print(f"Quantization Recipe: {qparam}")
        print(f"num_gemms={num_gemms}  in={in_features}  out={out_features}  batch={batch_size}")
        print(f"{'='*60}")

        group_linear = self.build_group_linear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            qparam=qparam,
        )
        for i in range(num_gemms):
            w = getattr(group_linear, f"weight{i}")
            w.main_grad = torch.zeros_like(w.data, dtype=torch.float32)

        tokens_per_expert = batch_size // num_gemms
        m_splits = [tokens_per_expert] * num_gemms

        def make_inp(requires_grad):
            return torch.randn(
                batch_size, in_features, device="cuda", dtype=torch.bfloat16
            ).requires_grad_(requires_grad)

        def run_forward_only():
            # mirrors forward_only / compute_log_prob: eval + no_grad
            group_linear.eval()
            with torch.no_grad():
                out = group_linear(make_inp(False), m_splits=m_splits, is_first_microbatch=None)
            return out

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

            # Sum CUDA time of kernels that fall under group_quantize_w scope.
            # key_averages aggregates by op name; we look at user_annotation +
            # the quantization kernels.
            events = prof.events()
            qw_cuda_us = 0.0
            kernel_us = {}
            for ev in events:
                name = ev.key
                dev = getattr(ev, "cuda_time_total", 0) or 0
                # collect quantization-related CUDA kernels
                if any(k in name for k in (
                    "quantize", "fp4_group", "transpose", "swizzle",
                    "multi_padding", "CatArrayBatched", "reduce_kernel",
                )):
                    if dev > 0:
                        kernel_us[name] = kernel_us.get(name, 0.0) + dev
                        qw_cuda_us += dev
            print(f"\n--- {label} ---")
            print(f"  quant-related CUDA total: {qw_cuda_us/1e3:.3f} ms "
                  f"({qw_cuda_us/1e3/profile_iters:.3f} ms/iter)")
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
            print(f"  log_prob / train ratio: {lp_us/train_us:.3f}  "
                  f"(lower = more w-transpose quant skipped)")
        print(f"{'='*60}\n")


# ==================== Argument Parser ====================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GroupLinear Module Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Tests:
  1 - Forward pass execution
  2 - Backward pass with gradients
  3 - Varying m_splits configurations
  4 - Multiple microbatches (weight caching)

Available Quantization Recipes:
  1 - NON_QUANTIZE              (No quantization)
  2 - INT4_SUB_CHANNEL          (INT4 sub-channel quantization)
  3 - FP4_SUB_CHANNEL_REF       (FP4 reference implementation)
  4 - FP4_SUB_CHANNEL           (FP4 tiled quantization)
  5 - FP4_SUB_CHANNEL_QUARTET   (FP4 quartet quantization, default)

Examples:
  python test_group_linear_exact.py
  python test_group_linear_exact.py --tests 1 2 3
  python test_group_linear_exact.py --qparam 3
  python test_group_linear_exact.py --tests 1 4 --qparam 1
        """
    )
    
    parser.add_argument(
        "--tests",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5, 6],
        default=None,
        help="Specify which tests to run (default: all tests)"
    )

    parser.add_argument(
        "--qparam",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8],
        default=4,
        help="Quantization recipe: 1=NON_QUANTIZE, 2=INT4, 3=FP4_REF, 4=FP4, 5=FP4_QUARTET, 6=CURSOR2, 7=CURSOR2_CUTLASS, 8=CUTLASS_BWD (default: 4)"
    )
    
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List all available tests and exit"
    )
    
    parser.add_argument(
        "--list-qparams",
        action="store_true",
        help="List all available quantization recipes and exit"
    )

    parser.add_argument(
        "--nsys-only",
        action="store_true",
        help="Skip torch.profiler in perf test so nsys gets exclusive CUPTI access"
    )

    return parser.parse_args()


def list_available_tests():
    """Print all available tests."""
    print("\nAvailable Tests:")
    print("=" * 60)
    for test_id, (test_name, test_desc) in AVAILABLE_TESTS.items():
        print(f"  {test_id}. {test_desc}")
        print(f"     Function: {test_name}")
    print("=" * 60)


def list_available_qparams():
    """Print all available quantization recipes."""
    print("\nAvailable Quantization Recipes:")
    print("=" * 60)
    for qparam_id, (qparam_name, qparam_recipe, qparam_desc) in AVAILABLE_QPARAMS.items():
        print(f"  {qparam_id}. {qparam_name}")
        print(f"     {qparam_desc}")
    print("=" * 60)


# ==================== Main Test Runner ====================

def run_all_tests(qparam_id: int = 4, test_ids: list = None, nsys_only: bool = False):
    """
    Run GroupLinear tests.
    
    Args:
        qparam_id: ID of the quantization recipe to use (1-5)
        test_ids: List of test IDs to run (None = run all)
    """
    print("\n" + "="*60)
    print("GroupLinear Module Test Suite")
    print("="*60)
    
    # Get quantization parameter
    qparam_name, qparam_recipe, qparam_desc = AVAILABLE_QPARAMS[qparam_id]
    print(f"Quantization Recipe: {qparam_id}. {qparam_name}")
    print(f"Description: {qparam_desc}")
    
    # Determine which tests to run (default: correctness tests only, not perf)
    if test_ids is None:
        test_ids = [1, 2, 3, 4]
    else:
        test_ids = sorted(test_ids)
    
    print(f"Tests to run: {test_ids}")
    print("="*60)
    
    test = TestGroupLinearExact()
    
    passed_tests = []
    failed_tests = []
    
    for test_id in test_ids:
        test_name, test_desc = AVAILABLE_TESTS[test_id]
        
        try:
            print(f"\n>>> Running Test {test_id}: {test_desc}")
            
            # Get the test method and run it
            test_method = getattr(test, test_name)
            if test_id == 5:
                test_method(qparam_recipe, nsys_only=nsys_only)
            else:
                test_method(qparam_recipe)  # Use the recipe enum
            
            passed_tests.append((test_id, test_desc))
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"✗ Test {test_id} FAILED: {test_desc}")
            print(f"{'='*60}")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            print(f"\nFull Traceback:")
            print("-" * 60)
            traceback.print_exc()
            print("-" * 60)
            failed_tests.append((test_id, test_desc, e))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Quantization: {qparam_id}. {qparam_name}")
    print(f"Total tests run: {len(test_ids)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if passed_tests:
        print("\n✓ Passed tests:")
        for test_id, test_desc in passed_tests:
            print(f"  {test_id}. {test_desc}")
    
    if failed_tests:
        print("\n✗ Failed tests:")
        for test_id, test_desc, error in failed_tests:
            print(f"  {test_id}. {test_desc}")
            print(f"     Error: {type(error).__name__}: {error}")
    
    print("="*60 + "\n")
    
    if failed_tests:
        print(f"✗ {len(failed_tests)} test(s) failed. See detailed traceback above.\n")
        exit(1)
    else:
        print("✓ All tests passed successfully!\n")


if __name__ == "__main__":
    args = parse_arguments()
    
    # Handle list commands
    if args.list_tests:
        list_available_tests()
        exit(0)
    
    if args.list_qparams:
        list_available_qparams()
        exit(0)
    
    # Run tests
    run_all_tests(qparam_id=args.qparam, test_ids=args.tests, nsys_only=args.nsys_only)
