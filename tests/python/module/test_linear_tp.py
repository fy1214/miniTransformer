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
            
        Returns:
            Linear layer
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        sequence_parallel = False
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
            qlinear_params
        )
        
        return linear
    
    def test_column_parallel_forward(self, qparam: QuantizeRecipe, tp_size: int = 2):
        """
        Test column parallel forward pass accuracy.
        
        Column parallel: split output dimension across TP ranks
        Output shape: [batch, out_features / tp_size]
        """
        print(f"\n{'='*60}")
        print(f"Testing Column Parallel Forward (tp_size={tp_size})")
        print(f"{'='*60}")
        
        in_features = 4096
        out_features = 4096  # Will be split across TP ranks
        batch_size = 1024
        seed = 42
        
        # Create TP group
        tp_group = create_tp_group(tp_size) if tp_size > 1 else None
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Build baseline (tp_size=1)
        linear_baseline = self.build_linear(
            in_features, out_features,
            parallel_mode='column',
            tp_size=1,
            tp_group=None,
            qparam=qparam,
            seed=seed
        )
        
        # Build TP version (tp_size>1)
        linear_tp = self.build_linear(
            in_features, out_features,
            parallel_mode='column',
            tp_size=tp_size,
            tp_group=tp_group,
            qparam=qparam,
            seed=seed
        )
        
        # Create input (same across all ranks)
        torch.manual_seed(seed)
        inp = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
        
        # Forward pass
        with torch.no_grad():
            out_baseline = linear_baseline(inp, is_first_microbatch=True)
            out_tp = linear_tp(inp, is_first_microbatch=True)
        
        if rank < tp_size:
            # For column parallel, each rank has out_features/tp_size outputs
            # Gather all ranks' outputs
            if tp_size > 1:
                # Gather outputs from all TP ranks
                out_tp_gathered = torch.empty(
                    batch_size, out_features,
                    device="cuda", dtype=out_tp.dtype
                )
                dist.all_gather_into_tensor(out_tp_gathered, out_tp, group=tp_group)
            else:
                out_tp_gathered = out_tp
            
            # Compare with baseline
            abs_diff = (out_baseline - out_tp_gathered).abs()
            rel_diff = abs_diff / (out_baseline.abs() + 1e-8)
            
            print(f"Rank {rank}:")
            print(f"  Baseline output shape: {out_baseline.shape}")
            print(f"  TP output shape (local): {out_tp.shape}")
            print(f"  TP output shape (gathered): {out_tp_gathered.shape}")
            print(f"  Max absolute diff: {abs_diff.max().item():.6e}")
            print(f"  Mean absolute diff: {abs_diff.mean().item():.6e}")
            print(f"  Max relative diff: {rel_diff.max().item():.6e}")
            print(f"  Mean relative diff: {rel_diff.mean().item():.6e}")
            
            # Assert accuracy
            torch.testing.assert_close(
                out_baseline, out_tp_gathered,
                rtol=1e-2, atol=1e-2,
                msg=f"Column parallel forward mismatch at rank {rank}"
            )
            
            print(f"✓ Column parallel forward pass accuracy verified!")
        
        print(f"{'='*60}\n")
    
    def test_row_parallel_forward(self, qparam: QuantizeRecipe, tp_size: int = 2):
        """
        Test row parallel forward pass accuracy.
        
        Row parallel: split input dimension across TP ranks
        Requires AllReduce to get final output
        """
        print(f"\n{'='*60}")
        print(f"Testing Row Parallel Forward (tp_size={tp_size})")
        print(f"{'='*60}")
        
        in_features = 4096  # Will be split across TP ranks
        out_features = 4096
        batch_size = 1024
        seed = 42
        
        # Create TP group
        tp_group = create_tp_group(tp_size) if tp_size > 1 else None
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Build baseline (tp_size=1)
        linear_baseline = self.build_linear(
            in_features, out_features,
            parallel_mode='row',
            tp_size=1,
            tp_group=None,
            qparam=qparam,
            seed=seed
        )
        
        # Build TP version (tp_size>1)
        linear_tp = self.build_linear(
            in_features, out_features,
            parallel_mode='row',
            tp_size=tp_size,
            tp_group=tp_group,
            qparam=qparam,
            seed=seed
        )
        
        # Create input (same across all ranks)
        torch.manual_seed(seed)
        inp = torch.randn(batch_size, in_features, device="cuda", dtype=torch.bfloat16)
        
        # Forward pass
        with torch.no_grad():
            out_baseline = linear_baseline(inp, is_first_microbatch=True)
            out_tp = linear_tp(inp, is_first_microbatch=True)
        
        if rank < tp_size:
            # Row parallel already includes AllReduce, so output shapes should match
            abs_diff = (out_baseline - out_tp).abs()
            rel_diff = abs_diff / (out_baseline.abs() + 1e-8)
            
            print(f"Rank {rank}:")
            print(f"  Baseline output shape: {out_baseline.shape}")
            print(f"  TP output shape: {out_tp.shape}")
            print(f"  Max absolute diff: {abs_diff.max().item():.6e}")
            print(f"  Mean absolute diff: {abs_diff.mean().item():.6e}")
            print(f"  Max relative diff: {rel_diff.max().item():.6e}")
            print(f"  Mean relative diff: {rel_diff.mean().item():.6e}")
            
            # Assert accuracy
            torch.testing.assert_close(
                out_baseline, out_tp,
                rtol=1e-2, atol=1e-2,
                msg=f"Row parallel forward mismatch at rank {rank}"
            )
            
            print(f"✓ Row parallel forward pass accuracy verified!")
        
        print(f"{'='*60}\n")
    
    def test_column_parallel_backward(self, qparam: QuantizeRecipe, tp_size: int = 2):
        """Test column parallel backward pass gradient accuracy."""
        print(f"\n{'='*60}")
        print(f"Testing Column Parallel Backward (tp_size={tp_size})")
        print(f"{'='*60}")
        
        in_features = 4096
        out_features = 4096
        batch_size = 1024
        seed = 42
        
        # Create TP group
        tp_group = create_tp_group(tp_size) if tp_size > 1 else None
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Build baseline
        linear_baseline = self.build_linear(
            in_features, out_features,
            parallel_mode='column',
            tp_size=1,
            tp_group=None,
            qparam=qparam,
            seed=seed
        )
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
            seed=seed
        )
        linear_tp.weight.main_grad = torch.zeros_like(
            linear_tp.weight.data, dtype=torch.float32
        )
        
        # Create input with gradients
        torch.manual_seed(seed)
        inp_baseline = torch.randn(
            batch_size, in_features, device="cuda", dtype=torch.bfloat16
        ).requires_grad_(True)
        
        torch.manual_seed(seed)
        inp_tp = torch.randn(
            batch_size, in_features, device="cuda", dtype=torch.bfloat16
        ).requires_grad_(True)
        
        # Forward and backward
        out_baseline = linear_baseline(inp_baseline, is_first_microbatch=True)
        out_tp = linear_tp(inp_tp, is_first_microbatch=True)
        
        out_baseline.sum().backward()
        out_tp.sum().backward()
        
        if rank < tp_size:
            # Compare input gradients
            if inp_baseline.grad is not None and inp_tp.grad is not None:
                abs_diff = (inp_baseline.grad - inp_tp.grad).abs()
                rel_diff = abs_diff / (inp_baseline.grad.abs() + 1e-8)
                
                print(f"Rank {rank} - Input Gradient:")
                print(f"  Max absolute diff: {abs_diff.max().item():.6e}")
                print(f"  Mean absolute diff: {abs_diff.mean().item():.6e}")
                print(f"  Max relative diff: {rel_diff.max().item():.6e}")
                
                torch.testing.assert_close(
                    inp_baseline.grad, inp_tp.grad,
                    rtol=1e-2, atol=1e-2,
                    msg=f"Column parallel input gradient mismatch at rank {rank}"
                )
            
            # Compare weight gradients
            if linear_baseline.weight.main_grad is not None and linear_tp.weight.main_grad is not None:
                # For column parallel, weight is split, need to gather
                if tp_size > 1:
                    weight_grad_gathered = torch.empty_like(linear_baseline.weight.main_grad)
                    dist.all_gather_into_tensor(
                        weight_grad_gathered, 
                        linear_tp.weight.main_grad,
                        group=tp_group
                    )
                else:
                    weight_grad_gathered = linear_tp.weight.main_grad
                
                abs_diff = (linear_baseline.weight.main_grad - weight_grad_gathered).abs()
                rel_diff = abs_diff / (linear_baseline.weight.main_grad.abs() + 1e-8)
                
                print(f"Rank {rank} - Weight Gradient:")
                print(f"  Max absolute diff: {abs_diff.max().item():.6e}")
                print(f"  Mean absolute diff: {abs_diff.mean().item():.6e}")
                
                torch.testing.assert_close(
                    linear_baseline.weight.main_grad, weight_grad_gathered,
                    rtol=1e-2, atol=1e-2,
                    msg=f"Column parallel weight gradient mismatch at rank {rank}"
                )
            
            print(f"✓ Column parallel backward pass accuracy verified!")
        
        print(f"{'='*60}\n")


# ==================== Main Test Runner ====================

def run_tp_tests(qparam_id: int = 5, tp_size: int = 2):
    """
    Run TP tests.
    
    Note: This requires running with torchrun:
    torchrun --nproc_per_node=2 test_linear_tp.py --qparam 5 --tp-size 2
    """
    # Map qparam_id to recipe
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
        print(f"Quantization: {qparam}")
        print(f"TP Size: {tp_size}")
        print("="*60)
    
    test = TestLinearTP()
    
    try:
        # Test 1: Column parallel forward
        test.test_column_parallel_forward(qparam, tp_size)
        
        # Test 2: Row parallel forward
        test.test_row_parallel_forward(qparam, tp_size)
        
        # Test 3: Column parallel backward
        test.test_column_parallel_backward(qparam, tp_size)
        
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
        "--local-rank",
        type=int,
        default=0,
        help="Local rank (set by torchrun)"
    )
    
    args = parser.parse_args()
    
    # Initialize distributed
    if 'RANK' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        initialize_distributed(rank, world_size)
    else:
        print("ERROR: This test requires running with torchrun!")
        print("Usage:")
        print("  torchrun --nproc_per_node=2 test_linear_tp.py --qparam 5 --tp-size 2")
        exit(1)
    
    try:
        run_tp_tests(qparam_id=args.qparam, tp_size=args.tp_size)
    finally:
        cleanup_distributed()
