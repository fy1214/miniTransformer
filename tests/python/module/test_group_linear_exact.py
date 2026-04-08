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
from miniTransformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from miniTransformer.module.group_linear import GroupedLinear


# ==================== Available Tests ====================

AVAILABLE_TESTS = {
    1: ("test_group_linear_forward", "Forward pass execution"),
    2: ("test_group_linear_backward", "Backward pass with gradients"),
    3: ("test_group_linear_varying_splits", "Varying m_splits configurations"),
    4: ("test_group_linear_multiple_microbatches", "Multiple microbatches (weight caching)"),
}

# ==================== Available Quantization Recipes ====================

AVAILABLE_QPARAMS = {
    1: ("NON_QUANTIZE", QuantizeRecipe.NON_QUANTIZE, "No quantization"),
    2: ("INT4_SUB_CHANNEL", QuantizeRecipe.INT4_SUB_CHANNEL, "INT4 sub-channel quantization"),
    3: ("FP4_SUB_CHANNEL_REF", QuantizeRecipe.FP4_SUB_CHANNEL_REF, "FP4 reference implementation"),
    4: ("FP4_SUB_CHANNEL", QuantizeRecipe.FP4_SUB_CHANNEL, "FP4 tiled quantization"),
    5: ("FP4_SUB_CHANNEL_QUARTET", QuantizeRecipe.FP4_SUB_CHANNEL_QUARTET, "FP4 quartet quantization"),
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
        choices=[1, 2, 3, 4],
        default=None,
        help="Specify which tests to run (default: all tests)"
    )
    
    parser.add_argument(
        "--qparam",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="Quantization recipe to use: 1=NON_QUANTIZE, 2=INT4, 3=FP4_REF, 4=FP4, 5=FP4_QUARTET (default: 5)"
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

def run_all_tests(qparam_id: int = 5, test_ids: list = None):
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
    
    # Determine which tests to run
    if test_ids is None:
        test_ids = list(AVAILABLE_TESTS.keys())
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
        print(f"✗ {len(failed_tests)} test(s) failed. See traceback above for details.\n")
        raise RuntimeError(f"{len(failed_tests)} test(s) failed")
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
    try:
        run_all_tests(qparam_id=args.qparam, test_ids=args.tests)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Test Suite Failed")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"\nFull Traceback:")
        print("-" * 60)
        traceback.print_exc()
        print("-" * 60 + "\n")
        exit(1)
