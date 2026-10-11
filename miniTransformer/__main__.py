"""Inspect recipes without importing the training implementation."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os

from .recipes import GROUPS, RECIPES, describe_recipe, get_recipe, recipe_summary


def check_recipe(recipe, environ=None) -> list[dict]:
    """Check selected runtime prerequisites without creating a model or tensors."""
    environ = os.environ if environ is None else environ
    spec = get_recipe(recipe)
    checks = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        add("PyTorch", "error", f"{exc}. Install CUDA-enabled PyTorch in the documented training environment.")
        return checks
    add("PyTorch", "ok", getattr(torch, "__version__", "installed"))
    try:
        if not torch.cuda.is_available():
            add("CUDA", "error", "No CUDA GPU visible. Use the target NVIDIA host/container and check GPU access.")
        else:
            device = torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(device)
            name = torch.cuda.get_device_name(device)
            requires_blackwell = spec.enum_value.startswith("fp4_")
            status = "error" if requires_blackwell and capability < (10, 0) else "ok"
            detail = f"{name}, compute capability {capability[0]}.{capability[1]}"
            if status == "error":
                detail += "; use the Blackwell environment for this recipe"
            add("GPU", status, detail)
    except Exception as exc:
        add("CUDA", "error", f"Cannot inspect GPU: {exc}. Check the NVIDIA driver and container GPU access.")

    try:
        te = importlib.import_module("transformer_engine.pytorch")
    except Exception as exc:
        add("Transformer Engine", "error", f"{exc}. Install/build TE as described in the environment setup; per-token recipes need PR #3045 interfaces.")
        return checks
    add("Transformer Engine", "ok", "PyTorch frontend imports successfully")
    if spec.enum_value.startswith("fp4_"):
        for name in ("NVFP4Quantizer", "MXFP8Quantizer"):
            exists = hasattr(te, name)
            add(name, "ok" if exists else "error",
                "available" if exists else f"Missing TE {name}; use the documented TE build.")

    if spec.qat_id in (15, 16, 17):
        quantizer = getattr(te, "NVFP4Quantizer", None)
        try:
            params = inspect.signature(quantizer.__init__).parameters if quantizer else {}
        except (TypeError, ValueError) as exc:
            add("Per-token quantizer", "error", f"Cannot inspect NVFP4Quantizer signature: {exc}")
            params = {}
        for name in ("per_token", "per_token_weight_2d"):
            exists = name in params
            add(name, "ok" if exists else "error",
                "available" if exists else f"Missing NVFP4Quantizer({name}=...); build TE with the PR #3045 weight-scaling interfaces.")
        try:
            tex = importlib.import_module("transformer_engine.pytorch.cpp_extensions")
            dense = all(hasattr(tex, name) for name in (
                "nvfp4_per_token_group_quantize_bulk_dense", "nvfp4_cutlass_grouped_per_token_gemm_dense"))
            listed = all(hasattr(tex, name) for name in (
                "nvfp4_per_token_group_quantize_bulk", "nvfp4_cutlass_grouped_per_token_gemm"))
            requested = environ.get("MINITE_NVFP4_PATH", "auto").strip().lower()
            if requested in ("dense", "1", "true", "on"):
                actual, available = "dense", dense
            elif requested in ("list", "legacy", "0", "false", "off"):
                actual, available = "list", listed
            else:
                actual, available = ("dense", dense) if dense else ("list", listed)
            add("Per-token layout", "ok" if available else "error",
                f"requested={requested}, resolved={actual}; " +
                ("quantization and GEMM bindings available" if available else
                 f"missing {actual} quantization/GEMM bindings; use the documented TE build or an available layout"))
        except Exception as exc:
            add("Per-token bindings", "error", f"Cannot inspect TE bindings: {exc}")

    if spec.operator:
        module, _, operator = spec.operator.rpartition(".")
        try:
            op_module = importlib.import_module(module)
            getattr(op_module, operator)
            add("Selected operator", "ok", spec.operator)
        except Exception as exc:
            add("Selected operator", "error", f"{type(exc).__name__}: {exc}. Check the selected recipe requirements and rebuild missing extensions in the training environment.")
    description = describe_recipe(spec, environ)
    for note in description["notes"]:
        add("Configuration", "warning", note)
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="miniTransformer recipe selection and environment checks")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("recipes", help="List common recipes; --all includes references and experiments")
    listing.add_argument("--all", action="store_true", help="Include all 18 recipes")
    description = commands.add_parser("describe", help="Explain a recipe and the current backward/layout overrides")
    description.add_argument("recipe", help="Public name, original enum value, or legacy QAT number")
    description.add_argument("--json", action="store_true", help="Print structured recipe metadata")
    doctor = commands.add_parser("doctor", help="Check prerequisites for one selected recipe")
    doctor.add_argument("--recipe", required=True, help="Public name, original enum value, or legacy QAT number")
    args = parser.parse_args(argv)
    if args.command == "recipes":
        for group in GROUPS if args.all else ("common",):
            print(f"{group}:")
            for spec in RECIPES:
                if spec.group == group:
                    qat = f"QAT{spec.qat_id}" if spec.qat_id is not None else "no QAT ID"
                    print(f"  {spec.name} ({qat})\n    {spec.description}")
        if not args.all:
            print("Use --all for reference and experimental recipes. Groups describe purpose, not validation status.")
        return 0
    try:
        spec = get_recipe(args.recipe)
        info = describe_recipe(spec)
        if args.command == "describe":
            if args.json:
                print(json.dumps(info, indent=2, ensure_ascii=False))
            else:
                print(recipe_summary(info))
                print(info["description"])
                print(f"Original enum value: {spec.enum_value}")
                print(f"Group: {spec.group}; operator: {spec.operator or 'quantization disabled'}")
                print("Requires: " + "; ".join(info["requirements"]))
                for note in info["notes"]:
                    print(f"Note: {note}")
                if info["environment"]:
                    print("Relevant environment:")
                    for name, value in info["environment"].items():
                        print(f"  {name}={value}")
            return 0
        print(recipe_summary(info))
        checks = check_recipe(args.recipe)
        for check in checks:
            print(f"[{check['status'].upper()}] {check['name']}: {check['detail']}")
        failed = any(check["status"] == "error" for check in checks)
        if not failed:
            print("Prerequisite checks passed. Numerical correctness and full training still require GPU validation.")
        return 1 if failed else 0
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
