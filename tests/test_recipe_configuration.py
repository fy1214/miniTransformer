"""CPU-only configuration and launcher contracts.

Run from the repository root:
    python -m unittest discover -s tests -p 'test_recipe_configuration.py' -v
No Torch, TE, pytest, model weights, or GPU is needed.
"""

import ast
from contextlib import redirect_stdout
import enum
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from miniTransformer import recipes
from miniTransformer.__main__ import check_recipe, main
from miniTransformer.utils.env import should_switch_to_hybrid_linear

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "miniTransformer/module/config.py"
LAUNCHER = ROOT / "develop/script/moe_training_e2e.sh"

# Frozen historical contract, rather than another lookup derived from the registry.
LEGACY = (
    "non_quantize", "int4_subchannel_scaling", "fp4_subchannel_scaling_ref",
    "fp4_subchannel_scaling", "fp4_subchannel_scaling_quartet",
    "fp4_subchannel_scaling_cursor2_ref", "fp4_subchannel_scaling_cursor2",
    "fp4_subchannel_scaling_cursor2_cutlass", "fp4_subchannel_scaling_w4a16",
    "fp4_qdq_block", "fp4_qdq_pertoken", "fp4_subchannel_scaling_cursor2_cutlass_bwd",
    "int4_marlin_forward", "fp4_subchannel_scaling_cursor2_cutlass_v2",
    "fp4_subchannel_scaling_pertoken", "fp4_subchannel_scaling_pertoken_w_pertensor",
    "fp4_subchannel_scaling_pertoken_bf16_bwd",
)


class RecipeTests(unittest.TestCase):
    def test_all_historical_ids_and_names(self):
        for index, value in enumerate(LEGACY, 1):
            with self.subTest(qat=index):
                spec = recipes.get_recipe(index)
                self.assertEqual(spec.enum_value, value)
                for selector in (str(index), spec.name, value, spec.name.upper()):
                    self.assertEqual(recipes.get_recipe(selector), spec)
                self.assertEqual(recipes.get_recipe_from_env({"QAT_PARAMS": str(index)})[0], spec)
                self.assertEqual(recipes.get_recipe_from_env({"MINITE_RECIPE": spec.name})[0], spec)

    def test_enum_and_reference_without_number(self):
        values = enum.Enum("Recipes", {f"R{i}": value for i, value in enumerate(LEGACY)})
        for member in values:
            self.assertEqual(recipes.get_recipe(member).enum_value, member.value)
        self.assertIsNone(recipes.get_recipe("bf16_ref").qat_id)
        self.assertEqual(len(recipes.RECIPES), 18)

    def test_registry_matches_runtime_enum(self):
        tree = ast.parse(CONFIG.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QuantizeRecipe")
        values = {n.value.value for n in node.body if isinstance(n, ast.Assign)}
        self.assertEqual(values, {r.enum_value for r in recipes.RECIPES})

    def test_default_empty_and_matching_selectors(self):
        for env in ({}, {"MINITE_RECIPE": "", "QAT_PARAMS": ""}, {"MINITE_RECIPE": "  ", "QAT_PARAMS": " "}):
            self.assertEqual(recipes.get_recipe_from_env(env), (recipes.get_recipe(6), "default"))
        self.assertEqual(recipes.get_recipe_from_env({"MINITE_RECIPE": "nvfp4_per_token", "QAT_PARAMS": "15"}),
                         (recipes.get_recipe(15), "MINITE_RECIPE"))

    def test_conflicts_and_invalid_selectors(self):
        with self.assertRaisesRegex(ValueError, "Conflicting selectors"):
            recipes.get_recipe_from_env({"MINITE_RECIPE": "nvfp4_per_token", "QAT_PARAMS": "17"})
        for value in (0, 18, -1, "", "unknown", True, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                recipes.get_recipe(value)
        for value in ("abc", "nvfp4_per_token", "1.5"):
            with self.assertRaisesRegex(ValueError, "integer"):
                recipes.get_recipe_from_env({"QAT_PARAMS": value})
        with self.assertRaisesRegex(ValueError, "Unknown recipe"):
            recipes.get_recipe_from_env({"MINITE_RECIPE": "misspelled", "QAT_PARAMS": "15"})

    def test_normalization_gate_parity(self):
        for index in range(1, 18):
            with patch.dict(os.environ, {"QAT_PARAMS": str(index)}, clear=True):
                self.assertTrue(should_switch_to_hybrid_linear())
            with patch.dict(os.environ, {"MINITE_RECIPE": recipes.get_recipe(index).name}, clear=True):
                self.assertTrue(should_switch_to_hybrid_linear())
        for env in ({}, {"QAT_PARAMS": "0"}):
            with patch.dict(os.environ, env, clear=True):
                self.assertFalse(should_switch_to_hybrid_linear())
        with patch.dict(os.environ, {"MINITE_RECIPE": "nvfp4_per_token", "QAT_PARAMS": "17"}, clear=True):
            # Norm selection does not re-resolve recipes: explicit layer params
            # can bypass environment selectors. The factory rejects conflicts.
            self.assertTrue(should_switch_to_hybrid_linear())
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                recipes.get_recipe_from_env()

    def test_backward_paths_remain_distinct(self):
        regular = recipes.describe_recipe(15, {})
        bypass = recipes.describe_recipe(15, {"DISABLE_BACKWARD_QUANT": "1"})
        dedicated = recipes.describe_recipe(17, {})
        self.assertEqual(regular["backward"], "MXFP8")
        self.assertEqual(bypass["backward_source"], "DISABLE_BACKWARD_QUANT")
        self.assertNotEqual(bypass["backward"], dedicated["backward"])
        overridden = recipes.describe_recipe(17, {"DISABLE_BACKWARD_QUANT": "1"})
        self.assertEqual(overridden["backward"], bypass["backward"])
        self.assertTrue(any("Unset DISABLE_BACKWARD_QUANT" in n for n in overridden["notes"]))
        self.assertEqual(recipes.describe_recipe(1, {"DISABLE_BACKWARD_QUANT": "1"})["backward_source"], "recipe")

    def test_description_does_not_change_environment(self):
        env = {"MINITE_NVFP4_PATH": "list", "DISABLE_BACKWARD_QUANT": "0"}
        before = dict(env)
        self.assertEqual(recipes.describe_recipe(15, env)["layout_requested"], "list")
        self.assertEqual(env, before)

    def test_factory_delegation_without_loading_gpu_operators(self):
        # Execute the actual public wrapper bodies with a recording factory. GPU
        # tests separately exercise the actual parameter/operator construction.
        tree = ast.parse(CONFIG.read_text())
        selected = [n for n in tree.body if
                    isinstance(n, ast.ClassDef) and n.name == "QuantizeRecipe" or
                    isinstance(n, ast.FunctionDef) and n.name in
                    ("get_qlinear_params", "get_qlinear_params_from_env", "get_qlinear_params_from_env_qat_params")]
        namespace = dict(enum=enum, Union=__import__("typing").Union, QLinearParams=object,
                         get_recipe=recipes.get_recipe, get_recipe_from_env=recipes.get_recipe_from_env,
                         _log_recipe_selection=lambda *a: None)
        calls = []
        def factory(recipe):
            calls.append(recipe.value)
            return SimpleNamespace(recipe=recipe.value, operator=object())
        namespace["get_qlinear_params_from_predefined"] = factory
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(CONFIG), "exec"), namespace)
        for index, value in enumerate(LEGACY, 1):
            a = namespace["get_qlinear_params"](index)
            b = namespace["get_qlinear_params"](recipes.get_recipe(index).name)
            self.assertEqual(a.recipe, value)
            self.assertEqual(b.recipe, value)
            self.assertIsNot(a, b)
            self.assertIsNot(a.operator, b.operator)
        with patch.dict(os.environ, {"MINITE_RECIPE": "nvfp4_per_token"}, clear=True):
            self.assertEqual(namespace["get_qlinear_params_from_env_qat_params"]().recipe, LEGACY[14])
        self.assertEqual(len(calls), 35)

    def test_optional_operator_imports_are_branch_local(self):
        tree = ast.parse(CONFIG.read_text())
        eager = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module and
                 n.module.startswith("miniTransformer.quantiztion.") and n.module.endswith("_op_pertoken")]
        self.assertEqual(eager, [])

    def test_smoke_script_keeps_legacy_test_ids_and_named_selection(self):
        path = ROOT / "tests/python/module/test_group_linear_exact.py"
        tree = ast.parse(path.read_text())
        parser = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parse_arguments")
        namespace = {"argparse": argparse}
        exec(compile(ast.Module(body=[parser], type_ignores=[]), str(path), "exec"), namespace)
        with patch.object(sys, "argv", [str(path), "--recipe", "nvfp4_per_token"]):
            args = namespace["parse_arguments"]()
            self.assertEqual(args.recipe, "nvfp4_per_token")
            self.assertIsNone(args.qparam)
        with patch.object(sys, "argv", [str(path), "--qparam", "9"]):
            self.assertEqual(namespace["parse_arguments"]().qparam, 9)
        mapping = next(n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "AVAILABLE_QPARAMS" for t in n.targets))
        expected = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7, 7: 8, 8: 12, 9: 15}
        for key, value in zip(mapping.value.keys, mapping.value.values):
            enum_member = value.elts[1].attr
            enum_class = next(n for n in ast.parse(CONFIG.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "QuantizeRecipe")
            enum_value = next(n.value.value for n in enum_class.body if isinstance(n, ast.Assign) and n.targets[0].id == enum_member)
            self.assertEqual(recipes.get_recipe(enum_value).qat_id, expected[key.value])

    def test_complete_guides_match_registry(self):
        for language in ("en", "zh"):
            content = (ROOT / f"develop/docs/{language}/recipes.md").read_text()
            for spec in recipes.RECIPES:
                self.assertIn(f"| `{spec.name}` | {spec.qat_id if spec.qat_id is not None else '—'} |", content)


class InspectionTests(unittest.TestCase):
    def cli(self, *args):
        env = dict(os.environ)
        for key in recipes.RECIPE_ENV_VARS:
            env.pop(key, None)
        # -S models a fresh environment with no installed GPU dependencies.
        return subprocess.run([sys.executable, "-B", "-S", "-m", "miniTransformer", *args],
                              cwd=ROOT, env=env, capture_output=True, text=True)

    def test_cli_without_gpu_dependencies(self):
        listing = self.cli("recipes")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertIn("nvfp4_per_token", listing.stdout)
        self.assertNotIn("(QAT16)", listing.stdout)
        all_recipes = self.cli("recipes", "--all")
        self.assertEqual(all_recipes.returncode, 0, all_recipes.stderr)
        self.assertIn("bf16_ref", all_recipes.stdout)
        self.assertIn("(QAT16)", all_recipes.stdout)
        description = self.cli("describe", "15", "--json")
        self.assertEqual(description.returncode, 0, description.stderr)
        self.assertEqual(json.loads(description.stdout)["name"], "nvfp4_per_token")
        doctor = self.cli("doctor", "--recipe", "nvfp4_per_token")
        self.assertEqual(doctor.returncode, 1)
        self.assertIn("[ERROR] PyTorch", doctor.stdout)
        self.assertNotIn("Traceback", doctor.stderr)

    def test_cli_unknown_recipe(self):
        result = self.cli("describe", "nvfp4_per_tokn")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Did you mean", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def fake_imports(self, dense=True, listed=True):
        class Quantizer:
            def __init__(self, per_token=False, per_token_weight_2d=False):
                pass
        cuda = SimpleNamespace(is_available=lambda: True, current_device=lambda: 0,
                               get_device_capability=lambda d: (10, 0), get_device_name=lambda d: "Test Blackwell")
        torch = SimpleNamespace(__version__="test", cuda=cuda)
        te = SimpleNamespace(NVFP4Quantizer=Quantizer, MXFP8Quantizer=object())
        bindings = {}
        if dense:
            bindings.update(nvfp4_per_token_group_quantize_bulk_dense=object(), nvfp4_cutlass_grouped_per_token_gemm_dense=object())
        if listed:
            bindings.update(nvfp4_per_token_group_quantize_bulk=object(), nvfp4_cutlass_grouped_per_token_gemm=object())
        tex = SimpleNamespace(**bindings)
        def importer(name):
            if name == "torch":
                return torch
            if name == "transformer_engine.pytorch":
                return te
            if name == "transformer_engine.pytorch.cpp_extensions":
                return tex
            return SimpleNamespace(Nvfp4TiledQuantizePertokenOp=object(), Nvfp4PertokenBf16BackwardOp=object())
        return importer

    def test_doctor_checks_selected_layout(self):
        with patch("miniTransformer.__main__.importlib.import_module", side_effect=self.fake_imports(dense=False)):
            checks = check_recipe(15, {"MINITE_NVFP4_PATH": "dense"})
            self.assertTrue(any(c["name"] == "Per-token layout" and c["status"] == "error" for c in checks))
            checks = check_recipe(15, {"MINITE_NVFP4_PATH": "auto"})
            self.assertFalse(any(c["status"] == "error" for c in checks))
            self.assertTrue(any("resolved=list" in c["detail"] for c in checks))
        with patch("miniTransformer.__main__.importlib.import_module", side_effect=self.fake_imports(listed=False)):
            checks = check_recipe(17, {"MINITE_NVFP4_PATH": "list"})
            self.assertTrue(any(c["status"] == "error" for c in checks))

    def test_doctor_pass_reports_only_prerequisites(self):
        output = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stdout(output), patch(
                "miniTransformer.__main__.importlib.import_module", side_effect=self.fake_imports()):
            self.assertEqual(main(["doctor", "--recipe", "15"]), 0)
        self.assertIn("Numerical correctness and full training still require GPU validation", output.getvalue())

    def test_doctor_reports_missing_per_token_weight_interface(self):
        importer = self.fake_imports()
        class OlderQuantizer:
            def __init__(self, per_token=False):
                pass
        def older_import(name):
            result = importer(name)
            if name == "transformer_engine.pytorch":
                result.NVFP4Quantizer = OlderQuantizer
            return result
        with patch("miniTransformer.__main__.importlib.import_module", side_effect=older_import):
            checks = check_recipe(15, {})
        self.assertTrue(any(c["name"] == "per_token_weight_2d" and c["status"] == "error" for c in checks))


class LauncherTests(unittest.TestCase):
    def launch(self, directory, backend, profile="off", extra_env=None, dry=True):
        env = dict(os.environ)
        env.update(LOG_ROOT=str(directory / "logs"), TRITON_CACHE_DIR=str(directory / "triton"),
                   NCCL_TMPDIR=str(directory / "nccl"), MEGATRON_ROOT=str(directory / "missing_megatron"),
                   MINITE_RECIPE="invalid_stale_selection", QAT_PARAMS="bad", TE_FP4_PERTOKEN="1",
                   DISABLE_BACKWARD_QUANT="1", PYTHONDONTWRITEBYTECODE="1")
        env.update(extra_env or {})
        args = ["bash", str(LAUNCHER), "--backend", backend, "--profile", profile]
        if dry:
            args.append("--dry-run")
        return subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)

    def test_new_and_old_backends_are_equivalent_and_dry_run_writes_nothing(self):
        pairs = (("qat14", "mini_nvfp4_cursor2_cutlass_v2"), ("qat15", "mini_nvfp4_per_token"),
                 ("qat16", "mini_nvfp4_per_token_w_per_tensor"), ("qat17", "mini_nvfp4_per_token_bf16_bwd"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for legacy, named in pairs:
                for profile in ("off", "nsys"):
                    with self.subTest(legacy=legacy, profile=profile):
                        a = self.launch(path, legacy, profile)
                        b = self.launch(path, named, profile)
                        self.assertEqual(a.returncode, 0, a.stderr)
                        self.assertEqual(b.returncode, 0, b.stderr)
                        self.assertEqual(a.stdout, b.stdout)
                        self.assertIn("command:", b.stdout)
            self.assertEqual(list(path.iterdir()), [])

    def test_help_and_missing_argument_have_no_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            env = dict(os.environ, LOG_ROOT=str(path / "logs"), TRITON_CACHE_DIR=str(path / "triton"),
                       NCCL_TMPDIR=str(path / "nccl"))
            for args, expected in ((["--help"], 0), (["--backend"], 2)):
                result = subprocess.run(["bash", str(LAUNCHER), *args], cwd=ROOT, env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)
            self.assertEqual(list(path.iterdir()), [])

    def test_native_and_ste_paths_clear_stale_selectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            native = self.launch(path, "te_bf16")
            self.assertEqual(native.returncode, 0, native.stderr)
            self.assertIn("MINITE_RECIPE=<unset>", native.stdout)
            bypass = self.launch(path, "qat15_disable_bwd")
            self.assertEqual(bypass.returncode, 0, bypass.stderr)
            self.assertIn("BF16 STE via layer bypass", bypass.stdout)
            dedicated = self.launch(path, "qat17")
            self.assertIn("BF16 dedicated operator", dedicated.stdout)
            self.assertIn("DISABLE_BACKWARD_QUANT=<unset>", dedicated.stdout)
            self.assertEqual(list(path.iterdir()), [])

    def test_metadata_serializes_recipe_and_quoted_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            bin_dir = path / "bin"
            bin_dir.mkdir()
            executable = bin_dir / "torchrun"
            executable.write_text("#!/bin/sh\nprintf 'stub training command\\n'\n")
            executable.chmod(0o755)
            exp = 'test "quoted" name'
            result = self.launch(path, "mini_nvfp4_per_token", extra_env={
                "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"], "EXP_NAME": exp,
            }, dry=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((path / "logs" / exp / "run_meta.json").read_text())
            self.assertEqual(metadata["exp_name"], exp)
            self.assertEqual(metadata["qat_id"], 15)
            self.assertEqual(metadata["recipe_name"], "nvfp4_per_token")
            self.assertEqual(metadata["recipe"]["backward_source"], "recipe")
            self.assertEqual(metadata["env"]["MINITE_RECIPE"], "nvfp4_per_token")
            self.assertEqual(metadata["env"]["DISABLE_BACKWARD_QUANT"], "")


if __name__ == "__main__":
    unittest.main()
