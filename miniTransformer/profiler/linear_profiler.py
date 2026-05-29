"""
Non-invasive torch.profiler wrapper for Linear / LayerNormLinear / GroupedLinear.

Activated via environment variables (see miniTransformer/utils/env.py).
Each module auto-attaches this profiler in __init__ when LINEAR_PROFILE_DIR is set.

Manual usage
------------
from miniTransformer.profiler import LinearProfiler

profiler = LinearProfiler(
    output_dir="/tmp/linear_profiles",
    num_captures=3,
    with_backward=True,
    rank=0,
)
profiler.attach(model.self_attention.query_key_value)
profiler.detach()  # optional early stop
"""

import os
import torch
from torch.profiler import profile, ProfilerActivity


# ---------------------------------------------------------------------------
# Tensor statistics helper
# ---------------------------------------------------------------------------

def _tensor_stats(t: torch.Tensor) -> dict:
    """Return scalar statistics for a tensor (moved to float32 for numerics)."""
    f = t.detach().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "min":   f.min().item(),
        "max":   f.max().item(),
        "mean":  f.mean().item(),
        "std":   f.std().item(),
        "norm":  f.norm().item(),
    }


def _fmt_stats(name: str, stats: dict) -> str:
    return (
        f"  {name:8s} shape={stats['shape']}  dtype={stats['dtype']}"
        f"  min={stats['min']:.4g}  max={stats['max']:.4g}"
        f"  mean={stats['mean']:.4g}  std={stats['std']:.4g}"
        f"  norm={stats['norm']:.4g}"
    )


class LinearStatsHook:
    """Lightweight hook that prints tensor statistics at four points per step:

        fwd_pre  — before forward  : inp, weight
        fwd_post — after  forward  : inp, weight, output
        bwd_pre  — before backward : dy
        bwd_post — after  backward : dy, dgrad, wgrad

    Activated via environment variables (see miniTransformer/utils/env.py):

        LINEAR_STATS_STEPS        : total steps to capture (required to enable; -1 = infinite)
        LINEAR_STATS_CLASS        : "linear" or "grouped" (default "linear")
        LINEAR_STATS_INSTANCE     : 0-based instance index (default 0)
        LINEAR_STATS_SKIP_STEPS   : forward calls to skip before printing (default 0)
        LINEAR_STATS_DUMP_DIR     : directory to save raw tensors as .pt files
        LINEAR_STATS_BWD          : 1 = also hook backward (default 1)
        LINEAR_STATS_RANK         : rank to collect on; -1 = all ranks (default 0)

    Parameters
    ----------
    label:         prefix used in log lines and file names.
    skip_steps:    forward calls to ignore before first capture.
    num_steps:     how many steps to record; -1 = no limit.
    with_backward: register bwd_pre and bwd_post hooks.
    dump_dir:      if not None, save tensors as <dump_dir>/<label>_step<N>_<name>.pt.
    rank:          local rank that performs capture; -1 = all ranks.
    """

    def __init__(
        self,
        label: str = "linear",
        skip_steps: int = 0,
        num_steps: int = 1,
        with_backward: bool = True,
        dump_dir: str | None = None,
        rank: int = 0,
    ):
        self.label = label
        self.skip_steps = skip_steps
        self.num_steps = num_steps
        self.with_backward = with_backward
        self.dump_dir = dump_dir
        self.rank = rank

        self._step_count = 0
        self._capture_count = 0
        self._handles: list = []
        self._module: torch.nn.Module | None = None
        self._module_attr: str | None = None
        # tensors stashed in fwd_pre for use in fwd_post and bwd hooks
        self._last_inp: torch.Tensor | None = None
        self._last_weight: torch.Tensor | None = None

    @property
    def done(self) -> bool:
        return self.num_steps >= 0 and self._capture_count >= self.num_steps

    def _should_capture(self) -> bool:
        if self.done:
            return False
        if self._step_count <= self.skip_steps:
            return False
        local_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        return self.rank < 0 or local_rank == self.rank

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, module: torch.nn.Module) -> "LinearStatsHook":
        self.detach()
        attr = f"_linear_stats_hook_{id(self)}"
        if hasattr(module, attr):
            return self  # already attached by this instance
        setattr(module, attr, self)
        self._module = module
        self._module_attr = attr
        handles = [
            module.register_forward_pre_hook(self._on_fwd_pre),
            module.register_forward_hook(self._on_fwd_post),
        ]
        if self.with_backward:
            handles.append(module.register_full_backward_pre_hook(self._on_bwd_pre))
            handles.append(module.register_full_backward_hook(self._on_bwd_post))
        self._handles = handles
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self._last_inp = None
        self._last_weight = None
        if self._module is not None:
            attr = getattr(self, "_module_attr", None)
            if attr and hasattr(self._module, attr):
                delattr(self._module, attr)
            self._module = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _on_fwd_pre(self, module, inputs):
        """Before forward: print inp, weight."""
        self._step_count += 1
        if not self._should_capture():
            return
        inp = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        weight = next((p for p in module.parameters(recurse=False)), None)
        self._last_inp = inp.detach()
        self._last_weight = weight.detach() if weight is not None else None

        step = self._capture_count
        lines = [f"[TensorStats] {self.label}  step={step}  fwd_pre"]
        lines.append(_fmt_stats("inp",    _tensor_stats(self._last_inp)))
        if self._last_weight is not None:
            lines.append(_fmt_stats("weight", _tensor_stats(self._last_weight)))
        print("\n".join(lines), flush=True)

    def _on_fwd_post(self, module, inputs, output):
        """After forward: print inp, weight, output."""
        if not self._should_capture():
            return
        step = self._capture_count
        lines = [f"[TensorStats] {self.label}  step={step}  fwd_post"]
        if self._last_inp is not None:
            lines.append(_fmt_stats("inp",    _tensor_stats(self._last_inp)))
        if self._last_weight is not None:
            lines.append(_fmt_stats("weight", _tensor_stats(self._last_weight)))
        out = output[0] if isinstance(output, (tuple, list)) else output
        if isinstance(out, torch.Tensor):
            lines.append(_fmt_stats("output",  _tensor_stats(out)))
        print("\n".join(lines), flush=True)
        if self.dump_dir is not None:
            self._dump(self._last_inp,    f"step{step:04d}_inp")
            self._dump(self._last_weight, f"step{step:04d}_weight")
            self._dump(out if isinstance(out, torch.Tensor) else None,
                       f"step{step:04d}_output")
        if not self.with_backward:
            self._capture_count += 1
            self._last_inp = None
            self._last_weight = None
            if self.done:
                self.detach()

    def _on_bwd_pre(self, module, grad_output):
        """Before backward: print dy (upstream gradient flowing in)."""
        if not self._should_capture():
            return
        step = self._capture_count
        dy = grad_output[0] if grad_output else None
        lines = [f"[TensorStats] {self.label}  step={step}  bwd_pre"]
        if dy is not None:
            lines.append(_fmt_stats("dy", _tensor_stats(dy)))
        print("\n".join(lines), flush=True)
        if self.dump_dir is not None:
            self._dump(dy, f"step{step:04d}_dy_pre")

    def _on_bwd_post(self, module, grad_input, grad_output):
        """After backward: print dy, dgrad, wgrad."""
        if not self._should_capture():
            return
        step = self._capture_count
        # grad_output[0] = dy  (upstream gradient, same as bwd_pre)
        # grad_input[0]  = dgrad (gradient w.r.t. module input)
        dy    = grad_output[0] if grad_output else None
        dgrad = grad_input[0]  if grad_input  else None
        lines = [f"[TensorStats] {self.label}  step={step}  bwd_post"]
        if dy is not None:
            lines.append(_fmt_stats("dy",    _tensor_stats(dy)))
        if dgrad is not None:
            lines.append(_fmt_stats("dgrad", _tensor_stats(dgrad)))
        weight = self._last_weight
        if weight is not None:
            wg = weight.grad
            if wg is not None:
                lines.append(_fmt_stats("wgrad",      _tensor_stats(wg)))
            elif hasattr(weight, "main_grad") and weight.main_grad is not None:
                lines.append(_fmt_stats("wgrad(main)", _tensor_stats(weight.main_grad)))
        print("\n".join(lines), flush=True)
        if self.dump_dir is not None:
            self._dump(dy,    f"step{step:04d}_dy")
            self._dump(dgrad, f"step{step:04d}_dgrad")
            if weight is not None and weight.grad is not None:
                self._dump(weight.grad, f"step{step:04d}_wgrad")
        self._capture_count += 1
        self._last_inp = None
        self._last_weight = None
        if self.done:
            self.detach()

    def _dump(self, t: torch.Tensor | None, name: str) -> None:
        if t is None:
            return
        os.makedirs(self.dump_dir, exist_ok=True)  # type: ignore[arg-type]
        path = os.path.join(self.dump_dir, f"{self.label}_{name}.pt")  # type: ignore[arg-type]
        torch.save(t.cpu(), path)
        print(f"[TensorStats] dumped → {path}", flush=True)


class LinearProfiler:
    """Attaches torch.profiler hooks to any nn.Module (Linear / LayerNormLinear / GroupedLinear).

    Parameters
    ----------
    output_dir:     directory where Chrome-trace .json files are written.
    num_captures:   number of traces to save before auto-detaching.
    skip_steps:     number of forward calls to skip before capturing (warmup).
    with_backward:  True  → each trace covers forward + backward.
                    False → each trace covers forward only.
    rank:           local rank that performs capture; -1 = all ranks.
    record_shapes:  forwarded to torch.profiler.profile.
    with_stack:     forwarded to torch.profiler.profile.
    label:          prefix for output file names (default "linear").
    """

    def __init__(
        self,
        output_dir: str,
        num_captures: int = 1,
        skip_steps: int = 10,
        with_backward: bool = True,
        rank: int = 0,
        record_shapes: bool = True,
        with_stack: bool = False,
        label: str = "linear",
    ):
        self.output_dir = output_dir
        self.num_captures = num_captures
        self.skip_steps = skip_steps
        self.with_backward = with_backward
        self.rank = rank
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.label = label

        self._capture_count = 0
        self._step_count = 0
        self._module: torch.nn.Module | None = None
        self._module_attr: str | None = None
        self._handles: list = []
        self._active_prof = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach(self, module: torch.nn.Module) -> "LinearProfiler":
        self.detach()
        attr = f"_linear_profiler_{id(self)}"
        if hasattr(module, attr):
            return self  # already attached by this instance
        setattr(module, attr, self)
        self._module = module
        self._module_attr = attr
        if self.with_backward:
            self._handles = [
                module.register_forward_hook(self._on_fwd_end),
                module.register_full_backward_hook(self._on_bwd_end),
            ]
        else:
            self._handles = [
                module.register_forward_pre_hook(self._on_fwd_start),
                module.register_forward_hook(self._on_fwd_end),
            ]
        return self

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        if self._active_prof is not None:
            try:
                self._active_prof.__exit__(None, None, None)
            except Exception:
                pass
            self._active_prof = None
        if self._module is not None:
            attr = getattr(self, "_module_attr", None)
            if attr and hasattr(self._module, attr):
                delattr(self._module, attr)
            self._module = None

    @property
    def done(self) -> bool:
        return self._capture_count >= self.num_captures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_capture(self) -> bool:
        if self.done:
            return False
        if self._step_count <= self.skip_steps:
            return False
        local_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        return self.rank < 0 or local_rank == self.rank

    def _open_prof(self) -> profile:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=self.record_shapes,
            profile_memory=True,
            with_stack=self.with_stack,
        )
        prof.__enter__()
        return prof

    def _close_and_save(self, prof: profile, direction: str) -> None:
        prof.__exit__(None, None, None)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(
            self.output_dir, f"{self.label}_{direction}_{self._capture_count:03d}.json"
        )
        prof.export_chrome_trace(path)
        print(f"[LinearProfiler] saved → {path}", flush=True)
        self._capture_count += 1
        if self.done:
            self.detach()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _on_fwd_start(self, module, inputs):
        """Pre-hook: start profiler before forward (forward-only mode)."""
        self._step_count += 1
        if self._should_capture():
            self._active_prof = self._open_prof()

    def _on_fwd_end(self, module, inputs, output):
        """Post-hook: fired after forward returns."""
        if self.with_backward:
            self._step_count += 1
            if self._should_capture():
                self._active_prof = self._open_prof()
        else:
            prof = self._active_prof
            if prof is not None:
                self._active_prof = None
                self._close_and_save(prof, "fwd")

    def _on_bwd_end(self, module, grad_input, grad_output):
        """Full-backward hook: close profiler after backward (with_backward mode)."""
        prof = self._active_prof
        if prof is not None:
            self._active_prof = None
            self._close_and_save(prof, "fwdbwd")
