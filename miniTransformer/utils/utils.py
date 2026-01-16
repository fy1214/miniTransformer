import torch
from contextlib import contextmanager
from typing import Callable, Tuple


HIGH_PRECISION_FLOAT_DTYPES = (
    torch.float,
    torch.float16,
    torch.bfloat16,
    torch.float32,
)

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def is_rank_0():
    return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def get_default_init_method() -> Callable[[torch.Tensor], torch.Tensor]:
    """Weight initialization method if not provided by user"""
    return init_method_normal(0.023)


def init_method_constant(val: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Init method to set all tensor elements to a constant value."""
    if val == 1.0:

        def init_(tensor: torch.Tensor) -> torch.Tensor:
            return torch.nn.init.ones_(tensor)

    elif val == 0.0:

        def init_(tensor: torch.Tensor) -> torch.Tensor:
            return torch.nn.init.zeros_(tensor)

    else:

        def init_(tensor: torch.Tensor) -> torch.Tensor:
            return torch.nn.init.constant_(tensor, val)

    return init_


def init_method_normal(sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Init method based on N(0, sigma)."""

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Cast tensor to dtype"""
    with torch.enable_grad():
        return tensor if tensor is None or tensor.dtype == dtype else tensor.to(dtype)


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    assert (
        numerator % denominator == 0
    ), f"{numerator} is not divisible by {denominator}"
    return numerator // denominator


def get_device_compute_capability() -> Tuple[int, int]:
    """CUDA compute capability of current GPU"""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return (props.major, props.minor)


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    # check device compute capability
    if get_device_compute_capability() >= (8, 9):  # ada = 8.9, hopper = 9.0
        return True, ""
    else:
        return False, "Device compute capability 8.9 or higher is required for FP8"


class FP8GlobalStateManager:
    """Global state manager for FP8 usage.

    Provides a way with identical signatures to TE to control FP8 behavior.
    """

    FP8_ENABLED = False
    FP8_PARAMETERS = False

    @classmethod
    def is_fp8_enabled(cls) -> bool:
        """Is FP8 enabled"""
        return cls.FP8_ENABLED

    @classmethod
    def is_fp8_params_enabled(cls) -> bool:
        """Should the parameters be stored as FP8"""
        return cls.FP8_PARAMETERS

    @classmethod
    def fp8_autocast_enter(
        cls,
        enabled: bool = False,
    ) -> None:
        """Set state and tracking variables for entry into FP8 region."""
        cls.FP8_ENABLED = enabled

        if enabled:
            fp8_available, reason_for_no_fp8 = check_fp8_support()
            assert fp8_available, reason_for_no_fp8

    @classmethod
    def set_fp8_autocast_state(cls, fp8_state: Tuple[bool]) -> None:
        """FP8 autocast state setter"""
        (cls.FP8_ENABLED,) = fp8_state

    @classmethod
    def get_fp8_autocast_state(cls) -> Tuple[bool]:
        """FP8 autocast state getter"""
        return (cls.FP8_ENABLED,)


@contextmanager
def fp8_autocast(
    enabled: bool = True,
):
    """
    Context manager for FP8 usage.
    """
    fp8_state = FP8GlobalStateManager.get_fp8_autocast_state()
    FP8GlobalStateManager.fp8_autocast_enter(enabled=enabled)
    try:
        yield
    finally:
        FP8GlobalStateManager.set_fp8_autocast_state(fp8_state)
