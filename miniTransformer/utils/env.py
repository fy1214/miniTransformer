import os

def should_switch_to_hybrid_linear():
    # Switch to kitchen linear if QAT_PARAMS is set
    qat_params_idx = int(os.getenv("QAT_PARAMS", "0"))
    should_switch = qat_params_idx > 0
    return should_switch


def get_keep_in_bf16_layer_numbers():
    keep_in_bf16_layer_numbers = os.getenv("QAT_KEEP_IN_BF16_LAYER_NUMBERS", None)
    if keep_in_bf16_layer_numbers is None:
        return []
    else:
        return [int(s) for s in keep_in_bf16_layer_numbers.split(",")]

def should_keep_backward_in_bf16():
    disable_backward_quant = int(os.getenv("DISABLE_BACKWARD_QUANT", "0"))
    return disable_backward_quant == 1