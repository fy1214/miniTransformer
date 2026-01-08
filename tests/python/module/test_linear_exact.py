import torch
from transformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from transformer.module.linear import Linear

class TestLinearExact:

    def build_linear(
        self, 
        in_features=None, 
        out_features=None
    ):
        in_features = in_features
        out_features = out_features
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
        parallel_mode = 'column'
        parameters_split = None
        device = "cuda"
        ub_overlap_rs = False
        ub_overlap_ag = False
        ub_name = None
        # JQ: hybrid additional args
        layer_number = 0
        qlinear_params = get_qlinear_params_from_predefined(QuantizeRecipe.FP4_SUB_CHANNEL_REF)

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
            # JQ: hybrid additional args
            layer_number,
            qlinear_params
        )
        return linear


    def test_linear_forward(self):
        linear = self.build_linear(1024, 4096)
        inp = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16)
        out = linear(inp, is_first_microbatch=True, fp8_output=False, inp_meta=None)

        print("Linear forward pass successful.")
    
    def test_linear_backward(self):
        linear = self.build_linear(1024, 4096)

        # mock main_grad
        linear.weight.main_grad = torch.zeros_like(linear.weight.data, dtype=torch.float32)
        
        inp = torch.randn(16, 1024, device="cuda", dtype=torch.bfloat16).requires_grad_(True)
        out = linear(inp, is_first_microbatch=True, fp8_output=False, inp_meta=None)

        # Compute gradients
        out.sum().backward()

        print("Linear backward pass successful.")

if __name__ == "__main__":
    test = TestLinearExact()
    test.test_linear_forward()
    test.test_linear_backward()