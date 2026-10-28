"""Tests for the logpdiff CLI and for backwards compatibility of the three
legacy scripts' --help entry points.

``logpdiff/cli.py`` / ``logpdiff/__main__.py`` are provided by a parallel
task; tests that need them skip cleanly when the modules are absent. The
legacy-script --help tests only need this repository and torch.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[1]

LEGACY_SCRIPTS = [
    "dump_expert_fc1_and_fc2_sglang.py",
    "dump_expert_fc1_and_fc2_megatron.py",
    "compare_expert_fc1_and_fc2.py",
]

SUBCOMMANDS = [
    "run",
    "collect-sglang",
    "collect-megatron",
    "compare",
    "report",
    "write-launch-meta",
    "export-routing",
    "sweep",
    "aggregate",
    "rl-metrics",
    "bootstrap",
]


def _require_cli():
    """Skip the calling test unless the parallel-task CLI module exists."""
    if not (TOOLS_DIR / "logpdiff" / "__main__.py").is_file():
        pytest.skip("logpdiff.__main__ is provided by a parallel task")
    if not (TOOLS_DIR / "logpdiff" / "cli.py").is_file():
        pytest.skip("logpdiff.cli is provided by a parallel task")


def _cli_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(TOOLS_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(*args, timeout=300):
    return subprocess.run(
        [sys.executable, "-m", "logpdiff", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(TOOLS_DIR),
        env=_cli_env(),
    )


def _write_config(tmp_path, config):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return str(path)


class TestLegacyScriptHelp:
    """--help must work without importing sglang/megatron/minite/etc."""

    @pytest.mark.parametrize("script", LEGACY_SCRIPTS)
    def test_help_exits_zero(self, script):
        proc = subprocess.run(
            [sys.executable, str(TOOLS_DIR / script), "--help"],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(TOOLS_DIR),
        )
        assert proc.returncode == 0, (
            f"{script} --help failed:\n{proc.stderr[-2000:]}"
        )
        assert "usage" in proc.stdout.lower()

    @pytest.mark.parametrize("script", LEGACY_SCRIPTS)
    def test_help_imports_no_heavy_dependencies(self, script):
        driver = Path(__file__).parent / "_run_help_guard.py"
        proc = subprocess.run(
            [sys.executable, str(driver), str(TOOLS_DIR / script)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(TOOLS_DIR),
        )
        assert proc.returncode == 0, (
            f"{script} --help triggered a heavy import or failed:\n"
            f"{proc.stderr[-2000:]}"
        )


class TestCliSmoke:
    def test_top_level_help(self):
        _require_cli()
        proc = _run_cli("--help")
        assert proc.returncode == 0, proc.stderr[-2000:]
        for sub in SUBCOMMANDS:
            assert sub in proc.stdout

    @pytest.mark.parametrize("sub", SUBCOMMANDS)
    def test_subcommand_help(self, sub):
        _require_cli()
        proc = _run_cli(sub, "--help")
        assert proc.returncode == 0, f"{sub} --help:\n{proc.stderr[-2000:]}"


class TestCliConfigPrecedence:
    """Merge order must be: built-in defaults < --config file < CLI flags."""

    def test_unknown_config_key_rejected(self, tmp_path, synth_run):
        _require_cli()
        cfg = _write_config(tmp_path, {"definitely_not_a_real_key": 1})
        proc = _run_cli("compare", "--config", cfg,
                        "--run-dir", str(synth_run.run_dir))
        assert proc.returncode != 0
        assert "definitely_not_a_real_key" in proc.stderr + proc.stdout

    def test_unknown_recipe_rejected(self, tmp_path, synth_run):
        _require_cli()
        help_text = _run_cli("compare", "--help").stdout
        if "recipe" not in help_text:
            pytest.skip("CLI exposes no --recipe option to probe")
        cfg = _write_config(tmp_path, {"recipe": "no-such-recipe"})
        proc = _run_cli("compare", "--config", cfg,
                        "--run-dir", str(synth_run.run_dir))
        assert proc.returncode != 0

    def test_cli_run_dir_beats_config(self, tmp_path, synth_run,
                                      synth_run_fail):
        _require_cli()
        # config points at the FAILING run, CLI at the PASSING one:
        # if CLI precedence works, strict mode sees the passing run -> exit 0
        cfg = _write_config(tmp_path, {"run_dir": str(synth_run_fail.run_dir)})
        proc = _run_cli("compare", "--config", cfg,
                        "--run-dir", str(synth_run.run_dir), "--strict")
        assert proc.returncode == 0, (
            "CLI --run-dir should override the config run_dir:\n"
            f"{proc.stderr[-2000:]}"
        )

    def test_config_plots_beat_defaults(self, tmp_path, synth_run):
        _require_cli()
        cfg = _write_config(tmp_path, {"plots": True})
        proc = _run_cli("report", "--config", cfg,
                        "--run-dir", str(synth_run.run_dir))
        assert proc.returncode == 0, proc.stderr[-2000:]
        report = synth_run.run_dir / "report.html"
        assert report.is_file()
        assert "<svg" in report.read_text(encoding="utf-8")


class TestCliStrictExitCodes:
    def test_pass_run_exits_zero_even_strict(self, synth_run):
        _require_cli()
        proc = _run_cli("compare", "--run-dir", str(synth_run.run_dir),
                        "--strict")
        assert proc.returncode == 0, proc.stderr[-2000:]

    def test_fail_run_exit_code_depends_on_strict(self, synth_run_fail):
        _require_cli()
        run_dir = str(synth_run_fail.run_dir)
        relaxed = _run_cli("compare", "--run-dir", run_dir)
        assert relaxed.returncode == 0, (
            "non-strict compare should report, not fail:\n"
            f"{relaxed.stderr[-2000:]}"
        )
        strict = _run_cli("compare", "--run-dir", run_dir, "--strict")
        assert strict.returncode != 0, (
            "--strict must turn a FAIL comparison into a non-zero exit code"
        )

    def test_compare_writes_metrics_artifacts(self, synth_run):
        _require_cli()
        proc = _run_cli("compare", "--run-dir", str(synth_run.run_dir))
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert (synth_run.run_dir / "metrics.json").is_file()


class TestMonitoringCommands:
    def test_write_launch_meta_command(self, synth_run):
        proc = _run_cli(
            "write-launch-meta",
            "--run-dir", str(synth_run.run_dir),
            "--checkpoint-step", "12",
            "--recipe", "nvfp4_te",
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        meta = json.loads(
            (synth_run.run_dir / "launch_meta.json").read_text(encoding="utf-8")
        )
        assert meta["checkpoint_step"] == 12
        assert meta["recipe"] == "nvfp4_te"

    def test_export_routing_command(self, tmp_path, synth_run):
        output = tmp_path / "routing.pt"
        proc = _run_cli(
            "export-routing",
            "--run-dir", str(synth_run.run_dir),
            "--output", str(output),
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert output.is_file()

    def test_rl_metrics_command(self, tmp_path, synth_run_fail):
        output = tmp_path / "rl.csv"
        proc = _run_cli(
            "rl-metrics",
            "--run-dir", str(synth_run_fail.run_dir),
            "--clip-epsilon", "0.2",
            "--output", str(output),
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert output.is_file()
        assert output.with_name("rl.summary.json").is_file()
