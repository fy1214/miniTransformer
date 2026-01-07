import torch
from transformer.module.config import QuantizeRecipe, get_qlinear_params_from_predefined
from transformer.module.linear import Linear

def check_linear_init():
    in_features = 1024
    out_features = 4096
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
    # JQ: hybrid addtional args
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

    print("Linear layer initialized successfully with quantization parameters.")

if __name__ == "__main__":
    check_linear_init()