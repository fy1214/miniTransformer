# Developer Guide

This guide covers how to install, integrate, and extend miniTransformer on **NVIDIA Blackwell (B200, sm_100+)**. Quantized operator paths are currently validated on Blackwell only.

| Language | Document |
|----------|----------|
| English (this page) | [`doc/develop/en/develop.md`](./develop.md) |
| 中文 | [`doc/develop/zh/develop.md`](../zh/develop.md) |

Related documentation:

- [Repository README](../../../README.md)
- [B200 patch notes](../../../patch/b200/README.md)
- [MoE e2e scripts](../../script/README.md)

---

## 1. Environment and Installation

### 1.1 Container image

Build the development image from [`patch/B300/dockerfile`](../../../patch/B300/dockerfile). The Dockerfile starts from a pinned SGLang base digest and installs Megatron-LM, a Transformer Engine (TE) fork, SGLang, slime, and miniTransformer. The build context can be empty (no local `COPY` dependencies).

**B200 (sm_100, default development path):**

```bash
cd /path/to/miniTransformer

DOCKER_BUILDKIT=1 docker build -f patch/B300/dockerfile \
  --build-arg FLASHINFER_CUDA_ARCH_LIST=10.0 \
  --build-arg FLASH_ATTN_CUDA_ARCH_LIST=10.0 \
  --build-arg NVTE_CUDA_ARCHS=100 \
  -t fy121415/dev-cu13-minite-b200:latest \
  .

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it fy121415/dev-cu13-minite-b200:latest /bin/bash
```

**B300 (sm_103) example:**

```bash
DOCKER_BUILDKIT=1 docker build -f patch/B300/dockerfile \
  --build-arg FLASHINFER_CUDA_ARCH_LIST=10.3 \
  --build-arg FLASH_ATTN_CUDA_ARCH_LIST=10.3 \
  --build-arg NVTE_CUDA_ARCHS=103 \
  -t fy121415/dev-cu13-minite-b300:latest \
  .
```

The default base image is `lmsysorg/sglang@sha256:42194170546745092e74cd5f81ad32a7c6e944c7111fe7bf13588152277ff356` (`v0.5.12-cu130`). Override with `--build-arg SGLANG_IMAGE=...`; prefer digests over floating tags. Additional build args (TE / Megatron / slime / miniTransformer branches, and others) are documented in the Dockerfile header comments.

Common in-container paths:

| Component | Path |
|-----------|------|
| miniTransformer | `/root/miniTransformer` |
| Megatron-LM | `/root/Megatron-LM` |
| Transformer Engine | `/root/TransformerEngine` (editable) |
| slime | `/root/slime` |
| SGLang | `/sgl-workspace/sglang` |

### 1.2 Develop / reinstall miniTransformer inside the image

The image already installs an editable miniTransformer from `MINITRANSFORMER_BRANCH` in the Dockerfile. To switch to a local workspace or a newer revision:

```bash
cd /root/miniTransformer
# After syncing to the target branch:
pip install -e . --no-build-isolation --no-deps
```

`pip install -e .` builds the C++/CUDA extension. After install, `import miniTransformer` works without injecting the package via `PYTHONPATH`.

### 1.3 Incremental patches (optional)

Images built from the Dockerfile already include the TE FP8-disable patch and patched Megatron / slime / SGLang branches. Running `patch/b200/setup.sh` is usually **not** required.

If you develop on an older `slimerl/slime:v0.2.3` image or another unpatched base, run:

```bash
cd /root/miniTransformer
bash patch/b200/setup.sh
```

The script is idempotent and applies:

| Target | Effect |
|--------|--------|
| Megatron-LM | MoE `align_size=128`; make `TEGroupedLinear` subclass miniTransformer `GroupedLinear` |
| Transformer Engine | Force dense Linear `fp8=False` under `--fp4` so non-expert layers stay BF16 |
| slime (`nvfp4` branch) | Always run `process_weights_after_loading` after weight updates |
| SGLang | Enable the per-token NVFP4 scale path (`SGLANG_NVFP4_PERTOKEN_SCALE`) |

Details and known issues: [patch/b200/README.md](../../../patch/b200/README.md).

---

## 2. Repository Layout

```
miniTransformer/
├── miniTransformer/
│   ├── module/           # Linear / LayerNormLinear / GroupedLinear
│   ├── quantiztion/      # QuantizeOp and QuantizeRecipe (note the directory spelling)
│   ├── ops/              # Python / CUDA / CuTeDSL / Triton kernel wrappers
│   ├── csrc/             # C++/CUDA extension and pybind
│   ├── distributed/      # Tensor-parallel, AllReduce, and related communication
│   ├── utils/env.py      # Env parsing (QAT, NVTE, DISABLE_BACKWARD_QUANT, …)
│   └── profiler/         # Linear-level profiling hooks
├── patch/b200/           # One-shot B200 patches
├── doc/script/           # MoE e2e, nsys, and sweep launchers
├── tests/python/         # Unit tests and fair benchmarks
└── tools/                # Diagnostics (e.g. logpdiff)
```

Call chain for MoE experts:

```
Megatron TEGroupedLinear
  → miniTransformer.GroupedLinear
    → QLinearParams (from QAT_PARAMS → QuantizeRecipe)
      → QuantizeOp.group_quantize / grouped_gemm
        → kernels under ops/ and csrc/
```

---

## 3. Training Integration

### 3.1 slime RL (Qwen3-30B-A3B)

See slime’s `scripts/run-qwen3-30B-A3B.sh`. Model flags live in `scripts/models/qwen3-30B-A3B.sh`.

When aligning with that training stack:

- Do **not** pass `--overlap-grad-reduce` or `--overlap-param-gather` (Megatron defaults are off; matches the slime scripts).
- Point `--ref-load` / `--load` at training checkpoints; use `--hf-checkpoint` for rollout.

NVFP4 end-to-end `PRECISE_ARGS` and Ray `env_vars` are documented in [patch/b200/README.md](../../../patch/b200/README.md).

### 3.2 Megatron MoE e2e

Use `doc/script/moe_training_e2e.sh` for multi-backend comparison. Keep `moe_training.sh` as the original TE-only launcher; do not layer miniTransformer logic onto it.

```bash
# BF16 smoke
bash develop/script/moe_training_bf16.sh

# Single backend
BACKEND=qat15 PROFILE=off TRAIN_ITERS=30 \
  bash develop/script/moe_training_e2e.sh

# Sweep
BACKENDS="te_bf16 te_nvfp4 te_nvfp4_pertoken qat14 qat15 qat15_disable_bwd qat16 qat17" \
  bash develop/script/moe_training_sweep.sh
```

`moe_training_e2e.sh` matches the slime Qwen3 settings (including omitting both DDP overlap flags). See [doc/script/README.md](../../script/README.md) for more detail.

### 3.3 Integration modes

| Mode | Typical use | Behavior |
|------|-------------|----------|
| Permanent patch (`patch/b200`) | slime / fixed images | `TEGroupedLinear` always subclasses miniTransformer `GroupedLinear` |
| Runtime swap (`SHOULD_REPLACE_TE_GROUPLINEAR=1`) | e2e / fair bench | Switch between native TE `GroupedLinear` and miniTransformer for `te_*` vs `qat*` A/B |

e2e sets `SHOULD_REPLACE_TE_GROUPLINEAR=1` and the matching `QAT_PARAMS` for `qat14`–`qat17`, and unsets the replace flag for `te_bf16` / `te_nvfp4*`.

For dense models, Megatron `TELinear` / `TELayerNormColumnParallelLinear` can subclass miniTransformer `Linear` / `LayerNormLinear` (see the repository README). The MoE path uses `GroupedLinear`.

---

## 4. Environment Variables

### 4.1 `QAT_PARAMS`

`Linear` and `GroupedLinear` resolve this variable through `get_qlinear_params_from_env_qat_params()` in `miniTransformer.module.config`. If unset, a warning is logged and index **6** (Cursor2 reference) is used.

Primary indices and e2e backends:

| `QAT_PARAMS` | Recipe | e2e backend | Notes |
|--------------|--------|-------------|-------|
| 14 | `FP4_SUB_CHANNEL_CURSOR2_CUTLASS_V2` | `qat14` | Cursor2 CUTLASS V2 |
| 15 | `FP4_SUB_CHANNEL_PERTOKEN` | `qat15` | Per-token activations; default `MINITE_NVFP4_PATH=dense` |
| 16 | `FP4_SUB_CHANNEL_PERTOKEN_W_PERTENSOR` | `qat16` | QAT15 activations + per-expert tensor weight amax |
| 17 | `FP4_SUB_CHANNEL_PERTOKEN_BF16_BWD` | `qat17` | Per-token forward; BF16 backward |

Indices 1–13 (INT4, quartet, legacy Cursor2, QDQ, and others) are defined in `get_qlinear_params_from_env_qat_params` in `config.py`.

> **Note:** Some older docs or scripts use `QAT_PARAM` (no trailing `S`). The runtime only reads `QAT_PARAMS`.

### 4.2 `SHOULD_REPLACE_TE_GROUPLINEAR`

| Value | Behavior |
|-------|----------|
| `1` | Use miniTransformer `GroupedLinear`; recipe selected by `QAT_PARAMS` |
| unset / other | Use native TE `GroupedLinear` (optionally with `--fp4-*` for TE NVFP4) |

### 4.3 Other common variables

| Variable | Meaning |
|----------|---------|
| `DISABLE_BACKWARD_QUANT=1` | Skip backward quantization (BF16 path); e2e backend `qat15_disable_bwd` |
| `MINITE_NVFP4_PATH` | Weight layout; default `dense` for QAT15/16 |
| `NVTE_NVFP4_DISABLE_RHT` | `0` enable RHT; `1` disable |
| `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING` | `0` enable SR; `1` disable |
| `NVTE_NVFP4_DISABLE_2D_QUANTIZATION` | `1` use 1D weight scales (aligns with SGLang `modelopt_fp4`) |
| `TE_FP4_PERTOKEN=1` | e2e `te_nvfp4_pertoken`: hooks switch Megatron to TE per-token recipe |
| `MOE_DUMP_DISPATCH=1` | Dump per-expert token histograms via e2e hooks |

Parsing lives in `miniTransformer/utils/env.py`.

---

## 5. Adding a QuantizeOp

Goal: register a new recipe selectable via `QAT_PARAMS=N` and invoked by `GroupedLinear` / `Linear`.

### 5.1 Implement the operator

Add a module under `miniTransformer/quantiztion/` that subclasses `QuantizeOpBase` (`quantization.py`) and implements:

| Method | Role |
|--------|------|
| `quantize(...)` | Dense `Linear` |
| `group_quantize(...)` | MoE `GroupedLinear` (via `m_splits`) |
| `qgemm(...)` | Dense GEMM |
| `grouped_gemm(...)` | MoE grouped GEMM (required for MoE) |
| `supports_allgather` | Whether quantize-then-all-gather is valid under TP; most NVFP4 ops return `False` |

Reference implementations:

- `nvfp4_quantize_op_pertoken.py` (QAT15)
- `nvfp4_cursor2_op_cutlass_v2.py` (QAT14)

Place kernels under `ops/` or `csrc/`; keep the Python op as orchestration and API adaptation.

### 5.2 Register recipe and index

Edit `miniTransformer/module/config.py`:

1. Add a `QuantizeRecipe` enum value.
2. Return a `QLinearParams` with `quantize_op=YourOp()` from `get_qlinear_params_from_predefined(...)`.
3. Add a `qat_params_idx == N` branch in `get_qlinear_params_from_env_qat_params()`.

Optionally add an e2e backend in `doc/script/moe_training_e2e.sh` that sets `SHOULD_REPLACE_TE_GROUPLINEAR=1` and `QAT_PARAMS=N`.

### 5.3 Smoke test

```bash
cd tests/python/module
python test_group_linear_exact.py --qparam 15 --tests 1 2

QAT_PARAMS=N python test_group_linear_exact.py --tests 1 2
```

Add precision coverage under `tests/python/nvfp4/` and e2e wiring under `doc/script/` as needed.

---

## 6. Testing

Commands assume the repository root, an editable install, and a Blackwell GPU.

### 6.1 NVFP4 (pytest)

```bash
pytest tests/python/nvfp4 -m "production and not bench" -v

pytest tests/python/nvfp4/quantize_nvfp4_pertoken/test_nvfp4_pertoken.py -v
```

Marker definitions (`production` / `legacy` / `eval` / `bench` / `slow`): [tests/python/nvfp4/README.md](../../../tests/python/nvfp4/README.md).

### 6.2 Module

```bash
python tests/python/module/test_group_linear_exact.py
python tests/python/module/test_group_linear_exact.py --qparam 14 --tests 1 2 3

python tests/python/module/test_group_linear_te_vs_mini.py
```

### 6.3 Megatron fair / low-bit

```bash
torchrun --nproc_per_node=4 \
  tests/python/megatron/bench_moe_pertoken_fair.py \
  --backend mini_qat15 --bench-mode fwb --preset qwen3-30b-a3b \
  --tp-size 4 --moe --moe-grouped-gemm --seq-length 4096 --num-layers 1

bash tests/python/megatron/run_fair_bench_te_vs_mini.sh
```

Heavier e2e and nsys flows live under `doc/script/`.

### 6.4 Distributed

See [tests/python/distributed/README.md](../../../tests/python/distributed/README.md).

---

## 7. New-Environment Checklist

1. Build and start the image from `patch/B300/dockerfile` (§1.1; B200 is the default path).
2. For local code changes, run `pip install -e . --no-build-isolation --no-deps` under `/root/miniTransformer`.
3. Run `bash patch/b200/setup.sh` only when not using the Dockerfile-based image.
4. Run `pytest tests/python/nvfp4 -m "production and not bench" -q`.
5. Run `BACKEND=te_bf16 PROFILE=off TRAIN_ITERS=5 bash doc/script/moe_training_bf16.sh`.
6. Optionally run `BACKEND=qat15 ... bash doc/script/moe_training_e2e.sh`.
7. For slime / e2e, confirm DDP overlap is not enabled (`--overlap-grad-reduce` / `--overlap-param-gather`).
