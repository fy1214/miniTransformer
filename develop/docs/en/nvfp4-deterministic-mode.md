# Qwen3-30B-A3B NVFP4 Deterministic Mode

中文版 / Chinese: [`nvfp4-deterministic-mode.md`](../zh/nvfp4-deterministic-mode.md)

This document describes how to prepare the runtime image, install dependencies inside the container, and launch the **NVFP4 deterministic** training and alignment workflow for Qwen3-30B-A3B.

It covers the slime **NVFP4 deterministic alignment stack**, which aligns Megatron training forward passes with SGLang rollout token log-probabilities. For the general NVFP4 QAT / TEGroupedLinear training path, see instead:

- [`doc/develop/zh/develop.md`](../zh/develop.md)
- [`patch/b200/README.md`](../../../patch/b200/README.md)

All filesystem locations are expressed as **environment variables**. They are not tied to any specific host or cluster layout. Export them according to your local environment before use.

---

## 1. Overview

| Item | Description |
|---|---|
| Model | Qwen3-30B-A3B (MoE); Hugging Face checkpoint in ModelOpt NVFP4 format (expert Linear layers quantized; attention, router, and LayerNorm typically remain BF16) |
| Alignment criterion | `train_rollout_logprob_abs_diff` below a configured threshold (default `MAX_TRAIN_ROLLOUT_DIFF=9.999e-7`) |
| Training path | Megatron + DeepEP + cutlass NVFP4 per-token MoE (forward); expert backward via BF16 recomputation |
| Rollout path | SGLang `modelopt_fp4` + `cutlass` + `SGLANG_NVFP4_PERTOKEN_SCALE=1` + FA4 + deterministic inference |
| Parallelism | Tensor Parallel = 1, Expert Tensor Parallel = 1, pure Expert Parallel (default `NUM_GPUS=8`). This alignment configuration does not support common production layouts such as TP=4 with EP=8 |
| Hardware | NVIDIA Blackwell (compute capability sm_100 or higher) is recommended; requires `fp4_gemm` from miniTransformer and FA4 |

This mode **cannot** be enabled on an arbitrary slime image by changing a few environment variables alone. The following must all be present:

1. A slime tree that includes the NVFP4 deterministic alignment implementation (commonly the `slime-deterministic-patch-nvfp4` branch, or an equivalent commit);
2. Deterministic and NVFP4 per-token patches applied to Megatron-LM and SGLang;
3. miniTransformer installed in the container such that `import fp4_gemm` succeeds.

---

## 2. Repositories, Branches, and Directory Layout

### 2.1 Build inputs on the host

| Component | Suggested variable | Notes |
|---|---|---|
| slime | `$SLIME_SRC` | Working tree that contains the NVFP4 deterministic alignment implementation |
| miniTransformer | `$MINITE_SRC` | Working tree that provides `fp4_gemm` and NVFP4 cutlass operators |
| Base image | `$BASE_IMAGE` | Prefer `slimerl/slime:latest-cu130` (CUDA 13 / Blackwell); `slimerl/slime:v0.2.3` with a compatible layout may be used if patches are applied afterward |

Point these variables at local checkouts or image tags. Do not rely on host-specific paths from internal clusters.

### 2.2 Default layout inside the container

The following defaults match the official slime image layout and may be overridden:

```bash
export SLIME_DIR="${SLIME_DIR:-/root/slime}"
export MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
export SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
export MINITE_DIR="${MINITE_DIR:-/root/miniTransformer}"
export PATCH_VERSION="${PATCH_VERSION:-latest}"   # or v0.5.15.post1
```

### 2.3 Checkpoint and dataset (required)

```bash
export HF_MODEL="${HF_MODEL:?set to the Qwen3-30B-A3B-NVFP4 Hugging Face checkpoint directory}"
export PROMPT_DATA="${PROMPT_DATA:?set to the prompt JSONL file (e.g. dapo-math-17k.jsonl)}"
export EVAL_DATA="${EVAL_DATA:-}"
export SAVE="${SAVE:-/tmp/qwen3_nvfp4_det_ckpt}"
export SGLANG_ROOT="${SGLANG_DIR}"
export MEGATRON_ROOT="${MEGATRON_DIR}"
```

---

## 3. Image Build and Environment Preparation

Two approaches are supported. **Approach A** is recommended: start from a published slime base image, then install patches and miniTransformer inside the container.

### 3.1 Approach A (recommended): base image + in-container setup

#### 3.1.1 Pull the base image and create a container

```bash
export BASE_IMAGE="${BASE_IMAGE:-slimerl/slime:latest-cu130}"
export CONTAINER_NAME="${CONTAINER_NAME:-slime-nvfp4-det}"

docker pull "${BASE_IMAGE}"

docker run -d --gpus all --name "${CONTAINER_NAME}" \
  --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${HF_MODEL}:${HF_MODEL}:ro" \
  -v "${PROMPT_DATA%/*}:${PROMPT_DATA%/*}:ro" \
  -v "${SLIME_SRC}:${SLIME_DIR}" \
  -v "${MINITE_SRC}:${MINITE_DIR}" \
  "${BASE_IMAGE}" sleep infinity

docker exec -it "${CONTAINER_NAME}" bash
```

Host checkouts may be bind-mounted into the container, or repositories may be cloned inside the container. The slime working tree must include the NVFP4 deterministic alignment implementation.

#### 3.1.2 Apply patches and install miniTransformer

Inside the container:

```bash
cd "${SLIME_DIR}"
bash scripts/setup-container-deterministic-patches.sh
```

This script will:

1. Apply to `$MEGATRON_DIR`  
   `docker/patch/$PATCH_VERSION/megatron.patch` and  
   `megatron-sglang-aligned.patch` (FP32 residual / fused RMSNorm alignment);
2. Apply available patches under `$SGLANG_DIR`, including  
   `sglang.patch`, `sglang-deterministic.patch`, `sglang-qwen3-deterministic.patch`,  
   and `sglang-nvfp4-pertoken.patch`;
3. Install miniTransformer from `$MINITE_DIR` with  
   `pip install -e . --no-deps --no-build-isolation`,  
   then verify `import fp4_gemm`.

Optional controls:

```bash
SKIP_PATCHES=1 bash scripts/setup-container-deterministic-patches.sh   # install miniTransformer only
SKIP_MINITE=1  bash scripts/setup-container-deterministic-patches.sh   # apply patches only
PATCH_VERSION=v0.5.15.post1 bash scripts/setup-container-deterministic-patches.sh
```

If the SGLang NVFP4 per-token implementation lives in a separate worktree:

```bash
export NVFP4_WORKTREE="${NVFP4_WORKTREE:?path to the SGLang NVFP4 per-token checkout}"
bash scripts/setup-sglang-nvfp4-pertoken.sh "${NVFP4_WORKTREE}"
```

#### 3.1.3 Commit a derived image (optional)

```bash
docker commit "${CONTAINER_NAME}" "${IMAGE_TAG:?e.g. registry.example.com/slime:nvfp4-det}"
docker push "${IMAGE_TAG}"   # if distribution is required
```

### 3.2 Approach B: rebuild the base image from slime `docker/Dockerfile`

Use this when a new CUDA 13 base image must be published.

```bash
cd "${SLIME_SRC}"
just release-cu13
```

Local build without push:

```bash
cd "${SLIME_SRC}"
docker build -f docker/Dockerfile . \
  --build-arg DEEPEP_CUDA_ARCH_LIST='10.0 10.3' \
  --build-arg SGLANG_IMAGE_TAG=v0.5.15.post1-cu130 \
  --build-arg PATCH_VERSION=latest \
  -t "${IMAGE_TAG:?registry.example.com/slime:cu130-local}"
```

Notes:

- The Dockerfile applies a subset of Megatron / SGLang patches. It does **not** guarantee that the full NVFP4 per-token and Qwen3 deterministic combination is included. After rebuilding, run  
  `scripts/setup-container-deterministic-patches.sh`, or ensure `sglang-nvfp4-pertoken.patch` is part of the image build.
- The slime application code must come from a commit or working tree that contains the alignment implementation; an unpatched upstream `main` branch is not a suitable runtime for this stack.

See `docker/README.md` and `docker/justfile` in the slime repository for base-image release rules.

---

## 4. Preflight Checks

After entering the container, verify the following in order.

### 4.1 Code and data

```bash
git -C "${SLIME_DIR}" branch --show-current
git -C "${MEGATRON_DIR}" log -1 --oneline
git -C "${SGLANG_DIR}" log -1 --oneline
test -f "${HF_MODEL}/config.json"
test -f "${PROMPT_DATA}"
```

### 4.2 NVFP4 operator stack

```bash
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_DIR}:${SGLANG_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"

python3 - <<'PY'
import fp4_gemm
from sglang.srt.layers.moe.cutlass_moe import (
    cutlass_moe_fp4_pertoken,
    cutlass_moe_fp4_pertoken_deepep_ll,
)
from sglang.srt.layers.quantization.fp4_utils import nvfp4_grouped_gemm, nvfp4_quantize_pertoken
from sglang.srt.server_args import ServerArgs
assert hasattr(fp4_gemm, "grouped_cutlass_gemm_v2")
assert callable(cutlass_moe_fp4_pertoken) and callable(cutlass_moe_fp4_pertoken_deepep_ll)
assert callable(nvfp4_grouped_gemm) and callable(nvfp4_quantize_pertoken)
assert "enable_fp32_moe_router" in ServerArgs.__dataclass_fields__
print("NVFP4 pertoken stack OK")
PY
```

### 4.3 Megatron alignment patch

```bash
grep -n '_use_sglang_fused_residual_rmsnorm' \
  "${MEGATRON_DIR}/megatron/core/transformer/transformer_layer.py"
```

A match indicates that `megatron-sglang-aligned.patch` is present.

### 4.4 Important environment variables

Training scripts export a full environment. The following variables have the largest impact on behavior:

| Variable | Typical value | Purpose |
|---|---|---|
| `SGLANG_NVFP4_PERTOKEN_SCALE` | `1` | Enable the cutlass MoE path with per-token activation scales |
| `MEGATRON_USE_SGLANG_FUSED_RESIDUAL_RMS` | `1` | Use SGLang-aligned FP32 residual RMSNorm in Megatron |
| `QWEN3_MOE_ALIGNED_SPEC` | `1` | Enable the `qwen3_moe_aligned` attention plugin |
| `SGLANG_DEEPGEMM_BATCH_INVARIANT` | `1` | DeepGEMM batch-invariant kernel selection |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | `64` | DeepEP low-latency padding limit; NVFP4 cutlass performs a fully padded GEMM over `[E, max_m]`, so large values reduce throughput substantially |
| `SGLANG_KV_CACHE_DTYPE` | `bfloat16` | KV-cache data type |
| `NUM_GPUS` / `CUDA_VISIBLE_DEVICES` | Must agree | Determines EP / DP / actor size; when both are set they must be equal |
| `MLP_SOCKET_IFNAME` | NIC name | Multi-node NCCL / NVSHMEM bootstrap; optional on a single node |

---

## 5. Execution

### 5.1 Alignment gate (run this first)

Use a short sequence length and a small batch to validate train / rollout log-probability alignment:

```bash
cd "${SLIME_DIR}"
ulimit -n 524288

export HF_MODEL PROMPT_DATA
export SGLANG_ROOT="${SGLANG_DIR}"
export MEGATRON_ROOT="${MEGATRON_DIR}"
export NUM_GPUS="${NUM_GPUS:-8}"
export MAX_TRAIN_ROLLOUT_DIFF="${MAX_TRAIN_ROLLOUT_DIFF:-9.999e-7}"
export SGLANG_NVFP4_PERTOKEN_SCALE=1
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_DIR}:${SGLANG_DIR}/python:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:${LD_LIBRARY_PATH:-}"

# Optional device pinning (must agree with NUM_GPUS):
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export NUM_GPUS=4

python3 -m pytest tests/test_qwen3_30b_a3b_nvfp4_deterministic_e2e.py -s -rs \
  --rollout-batch-size "${SLIME_E2E_ROLLOUT_BATCH_SIZE:-8}"
```

Success criteria: process exit code 0, and the log contains  
`Qwen3-30B-A3B NVFP4 deterministic train/rollout gate PASSED`.

Common failure modes:

| Symptom | Suggested action |
|---|---|
| `fp4_gemm` or `cutlass_moe_fp4_pertoken` import failure | Repeat the installation and verification steps in Sections 3 and 4 |
| Missing `_use_sglang_fused_residual_rmsnorm` in Megatron | Re-apply `megatron-sglang-aligned.patch` |
| Out of memory | Reduce `NUM_GPUS`, or lower `SLIME_E2E_ROLLOUT_BATCH_SIZE` / `SGLANG_MEM_FRACTION_STATIC` |

### 5.2 Full training script

Entry point:

```bash
cd "${SLIME_DIR}"
ulimit -n 524288

export HF_MODEL PROMPT_DATA SAVE
export REF_LOAD="${REF_LOAD:-${HF_MODEL}}"
export LOAD="${LOAD:-${HF_MODEL}}"
export SGLANG_ROOT="${SGLANG_DIR}"
export MEGATRON_ROOT="${MEGATRON_DIR}"
export NUM_GPUS="${NUM_GPUS:-8}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
export ENABLE_EVAL="${ENABLE_EVAL:-0}"
export USE_WANDB="${USE_WANDB:-0}"
export SGLANG_NVFP4_PERTOKEN_SCALE=1

bash scripts/run-qwen3-30B-A3B-nvfp4-deterministic.sh
```

Default workload shape (overridable via environment variables):

| Parameter | Default |
|---|---|
| `ROLLOUT_BATCH_SIZE` | 32 |
| `N_SAMPLES_PER_PROMPT` | 8 |
| `GLOBAL_BATCH_SIZE` | 256 |
| `ROLLOUT_MAX_RESPONSE_LEN` | 8192 |
| `ROLLOUT_MAX_CONTEXT_LEN` | 16384 |
| Reward | `--rm-type math` |
| SGLang | `modelopt_fp4` + `cutlass` + FA4 + DeepEP low-latency + deterministic inference |

Gate-sized workload (close to the e2e test):

```bash
SLIME_E2E_SHAPE=1 NUM_ROLLOUT=1 ENABLE_EVAL=0 CI_TEST=1 \
  bash scripts/run-qwen3-30B-A3B-nvfp4-deterministic.sh
```

`CI_TEST=1` enables  
`--ci-train-rollout-logprob-abs-diff-threshold ${MAX_TRAIN_ROLLOUT_DIFF}`.

Four-GPU example:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
SLIME_E2E_SHAPE=1 NUM_ROLLOUT=1 ENABLE_EVAL=0 CI_TEST=1 \
  bash scripts/run-qwen3-30B-A3B-nvfp4-deterministic.sh
```

### 5.3 Script execution summary

1. Terminate residual `sglang` / `ray` processes;
2. Export alignment environment variables (Section 4.4);
3. Start a `ray` head node;
4. Submit `python3 train.py` via `ray job submit`, with key flags including:
   - `--deterministic-mode`
   - `--megatron-cutlass-nvfp4-moe-forward-layers 0..47`
   - `--custom-megatron-before-*-hook-path ...nvfp4_alignment.enable_nvfp4_alignment_*`
   - `--sglang-quantization modelopt_fp4 --sglang-moe-runner-backend cutlass`
   - `--spec slime_plugins.models.qwen3_moe_aligned get_qwen3_moe_aligned_spec` (when `QWEN3_MOE_ALIGNED_SPEC=1`)

Metrics of interest:

- `train/train_rollout_logprob_abs_diff` for numerical alignment;
- `rollout/raw_reward`, entropy, and `grad_norm` for training behavior (independent of the alignment gate).

---

## 6. Troubleshooting

1. **`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` is too large**  
   On the DeepEP low-latency path, NVFP4 cutlass performs a fully padded GEMM over `[E, max_m]`. Inheriting a large value (for example 1024) from a serving script increases MoE cost substantially. Unless `SGLANG_DEEPEP_ALLOW_LARGE_MAX_M=1` is set, the training script clamps the value to 64.

2. **Tensor Parallel is not 1**  
   Alignment requires pure Expert Parallel. Do not retarget production TP=4 layouts in the parallelism arguments.

3. **`CUDA_VISIBLE_DEVICES` disagrees with `NUM_GPUS`**  
   The training script and e2e test exit with an error.

4. **miniTransformer installation mutates container dependencies**  
   Always use `pip install -e . --no-deps --no-build-isolation` so that PyTorch, Transformer Engine, and related packages in the image are left unchanged.

5. **Mixing with `patch/b200/setup.sh`**  
   `patch/b200` targets a different NVFP4 training recipe. Deterministic alignment should use  
   `setup-container-deterministic-patches.sh` and a slime tree that contains the alignment implementation.

6. **Confusing reward metrics with the alignment criterion**  
   Some Base checkpoints do not emit `</think>` markers, so deepscaler-style rewards may remain zero. The alignment gate must be judged by log-probability difference, not by reward.

7. **Process cleanup on shared nodes**  
   The training script may terminate existing `sglang` / `ray` processes at startup. On shared machines, confirm that this will not disrupt other jobs. If supported by the script version, set `RAY_SKIP_GLOBAL_CLEANUP=1` and manage cleanup explicitly.

---

## 7. Checklist

- [ ] `$BASE_IMAGE` is usable and GPUs are visible  
- [ ] `$SLIME_DIR` contains the NVFP4 deterministic alignment implementation  
- [ ] `setup-container-deterministic-patches.sh` completed successfully (or an equivalent patch + miniTransformer install)  
- [ ] Sections 4.2 and 4.3 pass  
- [ ] `$HF_MODEL` and `$PROMPT_DATA` exist and are readable  
- [ ] The alignment gate (e2e) passes  
- [ ] Launch the full-shape `run-qwen3-30B-A3B-nvfp4-deterministic.sh` workflow  

---

## 8. Related Files (slime repository)

| Path | Description |
|---|---|
| `scripts/run-qwen3-30B-A3B-nvfp4-deterministic.sh` | Full training / gate-shaped entry point |
| `tests/test_qwen3_30b_a3b_nvfp4_deterministic_e2e.py` | pytest alignment gate |
| `scripts/setup-container-deterministic-patches.sh` | Apply patches and install miniTransformer in-container |
| `scripts/setup-sglang-nvfp4-pertoken.sh` | Merge SGLang NVFP4 per-token support from a worktree |
| `docker/patch/latest/` (and `v0.5.15.post1/`) | Megatron / SGLang patch sources |
| `slime/backends/megatron_utils/alignment/nvfp4_alignment.py` | Megatron NVFP4 forward hooks |
| `slime/backends/megatron_utils/alignment/cutlass_nvfp4_moe_forward.py` | cutlass NVFP4 MoE implementation |
| `slime/backends/megatron_utils/alignment/residual_checkpoint.py` | Preserve FP32 residual across activation checkpointing |
| `docker/README.md`, `docker/justfile` | Base-image build and release |

This repository (miniTransformer) provides the `fp4_gemm` extension (`nvfp4_quantize_pertoken`, `grouped_cutlass_gemm_v2`, and related operators). Install it into the runtime container with  
`pip install -e . --no-deps --no-build-isolation`.
