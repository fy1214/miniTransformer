"""Run-directory writer for logpdiff.

Layout produced by :func:`create_run_dir`::

    <base_dir>/<run_id>/
        config.resolved.json
        manifest.json
        inputs.jsonl
        sglang/index.jsonl
        sglang/tensors/*.pt
        megatron/index.jsonl
        megatron/tensors/*.pt

SECURITY: ``.pt`` payload files written by this module are produced and
consumed only by logpdiff itself. Never point this tooling at external,
untrusted dump files, and always resolve payload paths through
:func:`assert_internal_payload` before loading, so a corrupted or malicious
index file cannot make the loader read outside the run directory.

Only the standard library is required at import time; torch is imported
lazily inside the functions that actually touch tensors or CUDA.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .schema import TensorRecord, check_single_device_parallel

FRAMEWORKS = ("sglang", "megatron")

# environment variables recorded into manifest.json
_ENV_EXACT = ("QAT_PARAMS", "SHOULD_REPLACE_TE_GROUPLINEAR")
_ENV_PREFIXES = ("MINITE_", "NVTE_NVFP4_", "SGLANG_NVFP4_")


# ---------------------------------------------------------------------------
# payload path safety
# ---------------------------------------------------------------------------


def assert_internal_payload(path, run_dir) -> Path:
    """Resolve ``path`` (relative paths resolve against ``run_dir``) and
    guarantee it stays inside ``run_dir``. Raises ValueError otherwise.

    Call this before every ``torch.load`` of a payload referenced by an
    index file.
    """
    root = Path(run_dir).resolve()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"payload path {p} escapes run directory {root}")
    return p


# ---------------------------------------------------------------------------
# run directory creation
# ---------------------------------------------------------------------------


def _default_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def create_run_dir(
    base_dir,
    run_id: str | None = None,
    *,
    model_path: str | None = None,
    tokenizer_path: str | None = None,
    seed: int | None = None,
    backend: str | None = None,
    tp: int = 1,
    ep: int = 1,
    pp: int = 1,
    force_routing: bool = False,
) -> "RunWriter":
    """Create ``<base_dir>/<run_id>/`` with sglang/ and megatron/ subdirs.

    ``run_id`` defaults to a UTC timestamp plus a short uuid. The directory
    must not already exist -- existing runs are never overwritten.

    The model/seed/parallelism/force_routing arguments are carried into the
    manifest written by :meth:`RunWriter.finalize_manifest`. P0 only
    supports TP=EP=PP=1.
    """
    check_single_device_parallel(tp, ep, pp)
    if run_id is None:
        run_id = _default_run_id()
    if (
        not run_id
        or run_id in (".", "..")
        or "/" in run_id
        or "\\" in run_id
        or os.sep in run_id
    ):
        raise ValueError(f"invalid run_id {run_id!r}")

    run_dir = Path(base_dir).resolve() / run_id
    if run_dir.exists():
        raise FileExistsError(
            f"run directory {run_dir} already exists; refusing to overwrite"
        )
    for framework in FRAMEWORKS:
        (run_dir / framework / "tensors").mkdir(parents=True, exist_ok=False)

    return RunWriter(
        run_dir=run_dir,
        run_id=run_id,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        seed=seed,
        backend=backend,
        tp=tp,
        ep=ep,
        pp=pp,
        force_routing=force_routing,
    )


# ---------------------------------------------------------------------------
# RunWriter
# ---------------------------------------------------------------------------


class RunWriter:
    """Appends records and payload tensors into one run directory."""

    def __init__(
        self,
        run_dir: Path,
        run_id: str,
        *,
        model_path: str | None = None,
        tokenizer_path: str | None = None,
        seed: int | None = None,
        backend: str | None = None,
        tp: int = 1,
        ep: int = 1,
        pp: int = 1,
        force_routing: bool = False,
    ) -> None:
        check_single_device_parallel(tp, ep, pp)
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.seed = seed
        self.backend = backend
        self.tp = tp
        self.ep = ep
        self.pp = pp
        self.force_routing = bool(force_routing)
        # weight_checksum -> run_dir-relative payload path of first occurrence
        self._weight_checksums: dict[str, str] = {}

    # -- small file helpers --------------------------------------------------

    def _framework_dir(self, framework: str) -> Path:
        if framework not in FRAMEWORKS:
            raise ValueError(
                f"unknown framework {framework!r}; must be one of {FRAMEWORKS}"
            )
        return self.run_dir / framework

    def _index_path(self, framework: str) -> Path:
        return self._framework_dir(framework) / "index.jsonl"

    def _append_jsonl(self, path: Path, obj: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_json(path: Path, obj) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    # -- top-level run files ---------------------------------------------------

    def write_config_resolved(self, config: dict) -> Path:
        """Write the fully-resolved run config to ``config.resolved.json``."""
        path = self.run_dir / "config.resolved.json"
        self._write_json(path, config)
        return path

    def write_inputs(self, records) -> Path:
        """Write input samples (one JSON object per line) to ``inputs.jsonl``."""
        path = self.run_dir / "inputs.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    # -- tensor records --------------------------------------------------------

    def append_record(self, framework: str, record: TensorRecord) -> None:
        """Validate ``record`` and append it to ``<framework>/index.jsonl``."""
        self._framework_dir(framework)  # validates framework name
        if record.framework != framework:
            raise ValueError(
                f"record.framework {record.framework!r} does not match "
                f"framework argument {framework!r}"
            )
        record.validate()
        self._append_jsonl(self._index_path(framework), record.to_dict())

    @staticmethod
    def _backfill_meta(record: TensorRecord, tensor) -> None:
        if not record.shape:
            record.shape = tuple(tensor.shape)
        if not record.dtype:
            record.dtype = str(tensor.dtype)

    def _save_tensor_file(self, framework: str, tensor) -> str:
        """Save ``tensor`` under ``<framework>/tensors/`` and return the
        run_dir-relative payload path (posix separators)."""
        torch = _import_torch()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected a torch.Tensor, got {type(tensor).__name__}")
        fname = f"{uuid.uuid4().hex}.pt"
        path = self._framework_dir(framework) / "tensors" / fname
        # store a detached CPU copy so payloads load on machines without CUDA
        torch.save(tensor.detach().cpu(), path)
        return (Path(framework) / "tensors" / fname).as_posix()

    def save_tensor(self, record: TensorRecord, tensor) -> str:
        """Save ``tensor``, backfill ``record.payload_path`` (and shape/dtype
        if unset), then append the record to the framework index.

        Returns the run_dir-relative payload path.
        """
        self._backfill_meta(record, tensor)
        record.payload_path = self._save_tensor_file(record.framework, tensor)
        self.append_record(record.framework, record)
        return record.payload_path

    def save_weight_tensor(self, record: TensorRecord, tensor) -> str:
        """Like :meth:`save_tensor`, but deduplicates by SHA-256 checksum.

        If an identical weight (same dtype, shape and bytes) was already
        stored in this run, no new file is written: ``record.payload_path``
        points at the existing payload and the index entry carries
        ``"weight_checksum"`` and ``"dedup_ref": true``. First occurrences
        also record ``"weight_checksum"`` in the index entry.
        """
        torch = _import_torch()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected a torch.Tensor, got {type(tensor).__name__}")
        self._backfill_meta(record, tensor)
        checksum = _tensor_sha256(tensor, torch)
        existing = self._weight_checksums.get(checksum)
        if existing is None:
            record.payload_path = self._save_tensor_file(record.framework, tensor)
            self._weight_checksums[checksum] = record.payload_path
            extra = {"weight_checksum": checksum}
        else:
            record.payload_path = existing
            extra = {"weight_checksum": checksum, "dedup_ref": True}
        if record.framework not in FRAMEWORKS:
            raise ValueError(
                f"unknown framework {record.framework!r}; must be one of {FRAMEWORKS}"
            )
        record.validate()
        entry = record.to_dict()
        entry.update(extra)
        self._append_jsonl(self._index_path(record.framework), entry)
        return record.payload_path

    # -- manifest ---------------------------------------------------------------

    def finalize_manifest(self, extra_env_allowlist=None) -> dict:
        """Write ``manifest.json`` and return its content as a dict.

        ``extra_env_allowlist`` is an optional iterable of additional
        environment variable names to record (values included).
        """
        manifest = {
            "run_id": self.run_id,
            "git_commit": _git_commit(),
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "versions": {
                "python": platform.python_version(),
                "torch": _get_version(("torch",), "torch"),
                "cuda": _cuda_version(),
                "transformer_engine": _get_version(
                    ("transformer-engine", "transformer_engine"), "transformer_engine"
                ),
                "sglang": _get_version(("sglang",), "sglang"),
                "megatron": _get_version(("megatron-core", "megatron"), "megatron"),
                "minite": _get_version(("minite",), "minite"),
            },
            "gpu_name": _gpu_name(),
            "env": _collect_env(extra_env_allowlist),
            "seed": self.seed,
            "backend": self.backend,
            "parallel": {"tp": self.tp, "ep": self.ep, "pp": self.pp},
            "force_routing": self.force_routing,
        }
        self._write_json(self.run_dir / "manifest.json", manifest)
        return manifest


# ---------------------------------------------------------------------------
# internal helpers (torch imported lazily)
# ---------------------------------------------------------------------------


def _import_torch():
    import torch  # noqa: F401 -- lazy: writer must import without torch

    return torch


def _tensor_sha256(tensor, torch) -> str:
    """SHA-256 over dtype, shape and raw bytes of a tensor."""
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(t.dtype).encode("utf-8"))
    h.update(repr(tuple(t.shape)).encode("utf-8"))
    if t.numel() > 0:
        flat = t.reshape(-1)
        # reinterpret as uint8 so dtype hasher also covers bf16/fp16 payload
        try:
            raw = flat.view(torch.uint8).numpy().tobytes()
        except (TypeError, RuntimeError):
            import io

            buf = io.BytesIO()
            torch.save(t, buf)
            raw = buf.getvalue()
        h.update(raw)
    return h.hexdigest()


def _get_version(dist_names, module_name: str):
    """Try importlib.metadata for each dist name, then module ``__version__``.

    Returns None when nothing can be determined.
    """
    for name in dist_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            break
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", None)
        if version is not None:
            return str(version)
    except Exception:
        pass
    return None


def _cuda_version():
    try:
        import torch

        return torch.version.cuda
    except Exception:
        return None


def _gpu_name():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def _git_commit():
    try:
        repo_root = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            commit = out.stdout.strip()
            return commit or None
    except Exception:
        pass
    return None


def _collect_env(extra_env_allowlist=None) -> dict:
    extra = set(extra_env_allowlist or ())
    env = {}
    for key, value in os.environ.items():
        if (
            key in _ENV_EXACT
            or key in extra
            or any(key.startswith(p) for p in _ENV_PREFIXES)
        ):
            env[key] = value
    return dict(sorted(env.items()))
