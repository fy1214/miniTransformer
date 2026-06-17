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


def should_disable_rht():
    disable_rht = os.getenv("NVTE_NVFP4_DISABLE_RHT", "0") == "1"
    return disable_rht


def should_disable_stochastic_rounding():
    disable_stochastic_rounding = (
        os.getenv("NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING", "0") == "1"
    )
    return disable_stochastic_rounding


def should_disable_2d_quantization():
    disable_2d_quantization= os.getenv("NVTE_NVFP4_DISABLE_2D_QUANTIZATION", "0") == "1"
    return disable_2d_quantization

    
def get_int4_real_group_size():
    return int(os.getenv("INT4_REAL_GROUP_SIZE", "128"))


def get_int4_fake_group_size():
    return int(
        os.getenv(
            "INT4_FAKE_GROUP_SIZE",
            os.getenv("OPEN_TRAINING_INT4_GROUP_SIZE", "16"),
        )
    )


def should_use_int4_real_symmetric():
    return os.getenv("INT4_REAL_SYMMETRIC", "1") == "1"

# ---------------------------------------------------------------------------
# Tensor statistics hook settings
# ---------------------------------------------------------------------------

def get_linear_tensor_stats_config():
    """Return stats-hook config dict if LINEAR_STATS_STEPS is set, else None.

    Environment variables
    ---------------------
    LINEAR_STATS_STEPS        : total steps to capture after skip (required to enable; -1 = infinite)
    LINEAR_STATS_CLASS        : "linear" or "grouped" (default "linear")
    LINEAR_STATS_INSTANCE     : 0-based index of the instance to attach (default 0)
    LINEAR_STATS_SKIP_STEPS   : forward calls to skip before printing (default 0)
    LINEAR_STATS_DUMP_DIR     : directory to dump raw tensors as .pt files; omit to skip dumping
    LINEAR_STATS_BWD          : 1 = also collect dy/dgrad/wgrad (default 1)
    LINEAR_STATS_RANK         : rank to collect on; -1 = all ranks (default 0)
    """
    num_steps_str = os.environ.get("LINEAR_STATS_STEPS", "")
    if not num_steps_str:
        return None
    target_class = os.environ.get("LINEAR_STATS_CLASS", "linear").lower()
    assert target_class in ("linear", "grouped"), (
        f"LINEAR_STATS_CLASS must be 'linear' or 'grouped', got '{target_class}'"
    )
    dump_dir = os.environ.get("LINEAR_STATS_DUMP_DIR", "") or None
    return dict(
        target_class=target_class,
        target_instance=int(os.environ.get("LINEAR_STATS_INSTANCE", "0")),
        num_steps=int(num_steps_str),
        skip_steps=int(os.environ.get("LINEAR_STATS_SKIP_STEPS", "0")),
        with_backward=os.environ.get("LINEAR_STATS_BWD", "1") == "1",
        dump_dir=dump_dir,
        rank=int(os.environ.get("LINEAR_STATS_RANK", "0")),
    )

def get_linear_profiler_config():
    """Return profiler config dict if LINEAR_PROFILE_DIR is set, else None.

    Environment variables
    ---------------------
    LINEAR_PROFILE_DIR        : output directory for Chrome-trace .json files (required to enable)
    LINEAR_PROFILE_CLASS      : which class to profile: "linear" (Linear/LayerNormLinear) or
                                "grouped" (GroupedLinear). Default "linear".
    LINEAR_PROFILE_INSTANCE   : 0-based index of the instance to attach within the target class
                                (default 0, i.e. first instance of that class).
    LINEAR_PROFILE_SKIP_STEPS : number of forward calls to skip before capturing (default 10)
    LINEAR_PROFILE_CAPTURES   : number of traces to save (default 1)
    LINEAR_PROFILE_BWD        : 1 = forward+backward per trace, 0 = forward only (default 1)
    LINEAR_PROFILE_RANK       : rank to capture on; -1 = all ranks (default 0)
    LINEAR_PROFILE_STACK      : 1 = record Python stack in trace (default 0)
    """
    output_dir = os.environ.get("LINEAR_PROFILE_DIR", "")
    if not output_dir:
        return None
    target_class = os.environ.get("LINEAR_PROFILE_CLASS", "linear").lower()
    assert target_class in ("linear", "grouped"), (
        f"LINEAR_PROFILE_CLASS must be 'linear' or 'grouped', got '{target_class}'"
    )
    return dict(
        output_dir=output_dir,
        target_class=target_class,
        target_instance=int(os.environ.get("LINEAR_PROFILE_INSTANCE", "0")),
        num_captures=int(os.environ.get("LINEAR_PROFILE_CAPTURES", "1")),
        skip_steps=int(os.environ.get("LINEAR_PROFILE_SKIP_STEPS", "10")),
        with_backward=os.environ.get("LINEAR_PROFILE_BWD", "1") == "1",
        rank=int(os.environ.get("LINEAR_PROFILE_RANK", "0")),
        with_stack=os.environ.get("LINEAR_PROFILE_STACK", "0") == "1",
    )
