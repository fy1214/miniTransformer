#!/usr/bin/env bash
# Qwen3-30B-A3B MoE e2e training perf launcher (TE vs miniTE backends).
#
# Backends (--backend / BACKEND):
#   te_bf16
#   te_nvfp4
#   te_nvfp4_pertoken
#   qat14 | mini_qat14
#   qat15 | mini_qat15
#   qat15_disable_bwd | mini_qat15_disable_bwd   # QAT15 + DISABLE_BACKWARD_QUANT=1
#   qat16 | mini_qat16   # weight per-expert tensor amax (W_PERTENSOR)
#   qat17 | mini_qat17   # former QAT16: per-token fwd + BF16 bwd
#
# Examples:
#   BACKEND=te_bf16 bash develop/script/moe_training_e2e.sh
#   bash develop/script/moe_training_e2e.sh --backend qat15 --profile nsys
#   BACKENDS="te_bf16 te_nvfp4 qat15" bash develop/script/moe_training_sweep.sh
#
# Note: leaves develop/script/moe_training.sh (original TE-only launcher) untouched.
#
# Env overrides (common):
#   MEGATRON_ROOT / MINITRANSFORMER_ROOT / CUDA_VISIBLE_DEVICES
#   NNODES / GPUS_PER_NODE / TP / PP / CP / EP
#   SEQ_LENGTH / TRAIN_ITERS / MICRO_BATCH_SIZE / GLOBAL_BATCH_SIZE
#   NUM_LAYERS / DATA_PATH / TOKENIZER_PATH / LOG_ROOT
#   PROFILE=off|nsys   (default nsys; torch profile via Megatron --profile*)
#   USE_MOCK_DATA=1    (default when DATA_PATH empty)
#   MINITE_NVFP4_PATH  (default dense for qat15/qat16)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINITRANSFORMER_ROOT="${MINITRANSFORMER_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
HOOKS_DIR="${SCRIPT_DIR}/moe_e2e_hooks"

NNODES="${NNODES:-1}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-localhost:29400}"
BACKEND="${BACKEND:-te_nvfp4_pertoken}"
PROFILE="${PROFILE:-nsys}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nnodes)
      NNODES="$2"
      shift 2
      ;;
    --rdzv_endpoint)
      RDZV_ENDPOINT="$2"
      shift 2
      ;;
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

normalize_backend() {
  local b
  b="$(echo "$1" | tr '[:upper:]' '[:lower:]')"
  case "$b" in
    te_bf16|te-bf16|bf16) echo "te_bf16" ;;
    te_nvfp4|te-nvfp4|nvfp4) echo "te_nvfp4" ;;
    te_nvfp4_pertoken|te-nvfp4-pertoken|pertoken) echo "te_nvfp4_pertoken" ;;
    qat14|mini_qat14) echo "qat14" ;;
    qat15|mini_qat15) echo "qat15" ;;
    qat15_disable_bwd|qat15-disable|mini_qat15_disable_bwd|qat15_disable) echo "qat15_disable_bwd" ;;
    qat16|mini_qat16) echo "qat16" ;;
    qat17|mini_qat17) echo "qat17" ;;
    *)
      echo "unknown backend: $1" >&2
      echo "expected: te_bf16 te_nvfp4 te_nvfp4_pertoken qat14 qat15 qat15_disable_bwd qat16 qat17" >&2
      exit 2
      ;;
  esac
}

BACKEND="$(normalize_backend "${BACKEND}")"

TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
EP="${EP:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
TRAIN_ITERS="${TRAIN_ITERS:-30}"
SAVE_ITERS="${SAVE_ITERS:-10000}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
NUM_LAYERS="${NUM_LAYERS:-48}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
MOE_TOPK="${MOE_TOPK:-8}"
MOE_FFN="${MOE_FFN:-768}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER:-root}/triton}"
export NCCL_TMPDIR="${NCCL_TMPDIR:-/tmp/${USER:-root}/nccl}"
export NVTE_NVFP4_DISABLE_RHT="${NVTE_NVFP4_DISABLE_RHT:-0}"
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING="${NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING:-0}"
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION="${NVTE_NVFP4_DISABLE_2D_QUANTIZATION:-1}"

mkdir -p "${TRITON_CACHE_DIR}" "${NCCL_TMPDIR}"

# Reset backend-specific env, then apply.
unset SHOULD_REPLACE_TE_GROUPLINEAR || true
unset QAT_PARAMS || true
unset DISABLE_BACKWARD_QUANT || true
unset TE_FP4_PERTOKEN || true
# Keep caller MINITE_NVFP4_PATH if set; otherwise set below for QAT15/16.

FP4_ARGS=()
QAT_BF16_LAYER_ARGS=()
# Megatron NVFP4 per-token amax-fuse (sort_chunks row_amax + skip-K1).
# Requires --moe-permute-fusion. Enabled for miniTE fuse-amax recipes.
AMAX_FUSE_ARGS=()
case "${BACKEND}" in
  te_bf16)
    ;;
  te_nvfp4)
    FP4_ARGS=(--fp4-format e2m1 --fp4-recipe nvfp4)
    ;;
  te_nvfp4_pertoken)
    export TE_FP4_PERTOKEN=1
    FP4_ARGS=(--fp4-format e2m1 --fp4-recipe nvfp4)
    ;;
  qat14)
    export SHOULD_REPLACE_TE_GROUPLINEAR=1
    export QAT_PARAMS=14
    FP4_ARGS=(--fp4-format e2m1 --fp4-recipe nvfp4)
    # QAT14 cursor2 op does not accept row_amax; amax-fuse is QAT15+.
    # Avoid TE layer autocast fighting miniTE GroupedLinear (see test_lowbit_param).
    QAT_BF16_LAYER_ARGS=(
      --first-last-layers-bf16
      --num-layers-at-start-in-bf16 "${NUM_LAYERS}"
      --num-layers-at-end-in-bf16 0
    )
    ;;
  qat15)
    export SHOULD_REPLACE_TE_GROUPLINEAR=1
    export QAT_PARAMS=15
    export MINITE_NVFP4_PATH="${MINITE_NVFP4_PATH:-dense}"
    FP4_ARGS=(--fp4-format e2m1 --fp4-recipe nvfp4)
    AMAX_FUSE_ARGS=(--moe-permute-fusion --nvfp4-pertoken-amax-fuse)
    QAT_BF16_LAYER_ARGS=(
      --first-last-layers-bf16
      --num-layers-at-start-in-bf16 "${NUM_LAYERS}"
      --num-layers-at-end-in-bf16 0
    )
    ;;
  qat15_disable_bwd)
    export SHOULD_REPLACE_TE_GROUPLINEAR=1
    export QAT_PARAMS=15
    export DISABLE_BACKWARD_QUANT=1
    export MINITE_NVFP4_PATH="${MINITE_NVFP4_PATH:-dense}"
    FP4_ARGS=(--fp4-format e2m1 --fp4-recipe nvfp4)
    AMAX_FUSE_ARGS=(--moe-permute-fusion --nvfp4-pertoken-amax-fuse)
    QAT_BF16_LAYER_ARGS=(
      --first-last-layers-bf16
      --num-layers-at-start-in-bf16 "${NUM_LAYERS}"
      --num-layers-at-end-in-bf16 0
    )
    ;;
  qat16)
    # QAT15 act per-token + weight per-expert tensor amax (broadcast into TE bulk).
    export SHOULD_REPLACE_TE_GROUPLINEAR=1
    export QAT_PARAMS=16
    export MINITE_NVFP4_PATH="${MINITE_NVFP4_PATH:-dense}"
    FP4_ARGS=(--fp4-format e2m1 --fp4-recipe nvfp4)
    AMAX_FUSE_ARGS=(--moe-permute-fusion --nvfp4-pertoken-amax-fuse)
    QAT_BF16_LAYER_ARGS=(
      --first-last-layers-bf16
      --num-layers-at-start-in-bf16 "${NUM_LAYERS}"
      --num-layers-at-end-in-bf16 0
    )
    ;;
  qat17)
    # Former QAT16: NVFP4 per-token forward + BF16 backward op.
    export SHOULD_REPLACE_TE_GROUPLINEAR=1
    export QAT_PARAMS=17
    export MINITE_NVFP4_PATH="${MINITE_NVFP4_PATH:-dense}"
    FP4_ARGS=(--fp4-format e2m1 --fp4-recipe nvfp4)
    AMAX_FUSE_ARGS=(--moe-permute-fusion --nvfp4-pertoken-amax-fuse)
    QAT_BF16_LAYER_ARGS=(
      --first-last-layers-bf16
      --num-layers-at-start-in-bf16 "${NUM_LAYERS}"
      --num-layers-at-end-in-bf16 0
    )
    ;;
esac

export PYTHONPATH="${HOOKS_DIR}:${MINITRANSFORMER_ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}"

LOG_ROOT="${LOG_ROOT:-/home/admin/log/exp}"
EXP_NAME="${EXP_NAME:-moe_e2e_${BACKEND}_N${NNODES}_g${GPUS_PER_NODE}_tp${TP}_pp${PP}_cp${CP}_ep${EP}}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${EXP_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${LOG_DIR}/tensorboard}"
DATA_PATH="${DATA_PATH:-}"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
USE_MOCK_DATA="${USE_MOCK_DATA:-}"
if [[ -z "${USE_MOCK_DATA}" ]]; then
  if [[ -z "${DATA_PATH}" ]]; then
    USE_MOCK_DATA=1
  else
    USE_MOCK_DATA=0
  fi
fi

mkdir -p "${LOG_DIR}" "${TENSORBOARD_DIR}"
META_JSON="${LOG_DIR}/run_meta.json"

cat >"${META_JSON}" <<EOF
{
  "backend": "${BACKEND}",
  "exp_name": "${EXP_NAME}",
  "megatron_root": "${MEGATRON_ROOT}",
  "minitransformer_root": "${MINITRANSFORMER_ROOT}",
  "tp": ${TP}, "pp": ${PP}, "cp": ${CP}, "ep": ${EP},
  "gpus_per_node": ${GPUS_PER_NODE}, "nnodes": ${NNODES},
  "seq_length": ${SEQ_LENGTH}, "train_iters": ${TRAIN_ITERS},
  "micro_batch_size": ${MICRO_BATCH_SIZE}, "global_batch_size": ${GLOBAL_BATCH_SIZE},
  "num_layers": ${NUM_LAYERS}, "num_experts": ${NUM_EXPERTS},
  "profile": "${PROFILE}",
  "use_mock_data": ${USE_MOCK_DATA},
  "env": {
    "SHOULD_REPLACE_TE_GROUPLINEAR": "${SHOULD_REPLACE_TE_GROUPLINEAR:-}",
    "QAT_PARAMS": "${QAT_PARAMS:-}",
    "DISABLE_BACKWARD_QUANT": "${DISABLE_BACKWARD_QUANT:-}",
    "TE_FP4_PERTOKEN": "${TE_FP4_PERTOKEN:-}",
    "MINITE_NVFP4_PATH": "${MINITE_NVFP4_PATH:-}",
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "${NVTE_NVFP4_DISABLE_2D_QUANTIZATION}"
  }
}
EOF

echo "============================================================"
echo "MoE e2e perf  backend=${BACKEND}  profile=${PROFILE}"
echo "EXP_NAME=${EXP_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "MEGATRON_ROOT=${MEGATRON_ROOT}"
echo "mock_data=${USE_MOCK_DATA}  train_iters=${TRAIN_ITERS}  seq=${SEQ_LENGTH}"
echo "parallel: tp=${TP} pp=${PP} cp=${CP} ep=${EP} gpus/node=${GPUS_PER_NODE}"
echo "env: REPLACE=${SHOULD_REPLACE_TE_GROUPLINEAR:-} QAT=${QAT_PARAMS:-} DISABLE_BWD=${DISABLE_BACKWARD_QUANT:-} TE_PERTOKEN=${TE_FP4_PERTOKEN:-} NVFP4_PATH=${MINITE_NVFP4_PATH:-}"
echo "============================================================"

DISTRIBUTED_ARGS=(
  --nnodes="${NNODES}"
  --nproc_per_node="${GPUS_PER_NODE}"
  --rdzv_endpoint="${RDZV_ENDPOINT}"
  --rdzv_backend=c10d
)

OPTIMIZE_ARGS=(
  --use-flash-attn
  --use-rotary-position-embeddings
  --no-masked-softmax-fusion
  --use-distributed-optimizer
  --bf16
  --accumulate-allreduce-grads-in-fp32
  # Container APEX often lacks fused_weight_gradient_mlp_cuda (ABI mismatch).
  --no-gradient-accumulation-fusion
)

TRAIN_ARGS=(
  --seed 1236
  --lr 3e-4
  --lr-decay-style cosine
  --min-lr 1e-5
  --weight-decay 1e-1
  --lr-warmup-iters 0
  --attention-dropout 0.0
  --init-method-std 0.02
  --hidden-dropout 0.0
  --clip-grad 1.0
  --adam-beta1 0.9
  --adam-beta2 0.95
  --initial-loss-scale 4096
  --bf16
  --train-iters "${TRAIN_ITERS}"
  --seq-length "${SEQ_LENGTH}"
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size "${TP}"
  --pipeline-model-parallel-size "${PP}"
  --context-parallel-size "${CP}"
  --expert-model-parallel-size "${EP}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  # No --overlap-grad-reduce / --overlap-param-gather: matches slime Qwen3
  # training, and avoids mcore DDP assert when output_layer is untied
  # (single-param bucket > default 40M elements).
)

GPT_ARGS=(
  # Qwen3-30B-A3B stores embedding and output_layer as two tensors (same as
  # slime scripts/models/qwen3-30B-A3B.sh). Without this flag a tied graph
  # silently drops output_layer.weight under assume_ok_unexpected.
  --untie-embeddings-and-output-weights
  --qk-layernorm
  --max-position-embeddings "${SEQ_LENGTH}"
  --num-layers "${NUM_LAYERS}"
  --hidden-size "${HIDDEN_SIZE}"
  --ffn-hidden-size "${MOE_FFN}"
  --num-attention-heads 32
  --kv-channels 128
  --rotary-base 1000000
  --disable-bias-linear
  --position-embedding-type rope
  --normalization RMSNorm
  --swiglu
  --attention-softmax-in-fp32
  --group-query-attention
  --num-query-groups 4
  --norm-epsilon 1e-6
)

MOE_ARGS=(
  --num-experts "${NUM_EXPERTS}"
  --moe-router-topk "${MOE_TOPK}"
  --moe-ffn-hidden-size "${MOE_FFN}"
  --moe-grouped-gemm
  --moe-aux-loss-coeff 0.001
  --moe-token-dispatcher-type alltoall
  "${AMAX_FUSE_ARGS[@]}"
)

# Qwen3 vocab embedding size. NullTokenizer treats --vocab-size as *without* EOD
# and exposes tokenizer.vocab_size = vocab_size + 1 with eod == vocab_size, so mock
# must pass (padded - 1) or embedding gather OOBs on the EOD id.
PADDED_VOCAB_SIZE="${PADDED_VOCAB_SIZE:-151936}"

DATA_ARGS=(
  --split 90,5,5
  --no-create-attention-mask-in-dataloader
  --num-workers "${NUM_WORKERS:-2}"
)
if [[ "${USE_MOCK_DATA}" == "1" ]]; then
  # NullTokenizer(vocab_size=N) => eod=N, tokenizer.vocab_size=N+1
  NULL_VOCAB_SIZE="${VOCAB_SIZE:-$((PADDED_VOCAB_SIZE - 1))}"
  DATA_ARGS+=(
    --mock-data
    --tokenizer-type NullTokenizer
    --vocab-size "${NULL_VOCAB_SIZE}"
  )
else
  DATA_ARGS+=(
    --data-path "${DATA_PATH}"
    --data-cache-path "${DATA_CACHE_PATH}"
    --tokenizer-model "${TOKENIZER_PATH}"
    --tokenizer-type HuggingFaceTokenizer
  )
fi

OUTPUT_ARGS=(
  --log-interval 1
  --save-interval "${SAVE_ITERS}"
  --eval-interval 10000
  --eval-iters 0
  --no-save-optim
  --no-load-optim
  --no-load-rng
  --log-throughput
  --log-timers-to-tensorboard
  --log-params-norm
  --log-num-zeros-in-grad
  --log-memory-to-tensorboard
  --tensorboard-dir "${TENSORBOARD_DIR}"
  --wandb-project "${WANDB_PROJECT:-moe_e2e_perf}"
  --wandb-exp-name "${EXP_NAME}"
  --wandb-save-dir "${LOG_DIR}"
)
# Optional checkpoint IO (disabled by default for short perf runs).
if [[ "${SAVE_CKPT:-0}" == "1" ]]; then
  OUTPUT_ARGS+=(--save "${LOG_DIR}/checkpoints" --load "${LOG_DIR}/checkpoints")
fi

PROFILER_ARGS=()
if [[ "${PROFILE}" != "off" ]]; then
  PROFILER_ARGS=(
    --profile
    --profile-step-start "${PROFILE_STEP_START:-20}"
    --profile-step-end "${PROFILE_STEP_END:-24}"
    --profile-ranks 0
  )
fi

INFIX_ARGS=(
  --padded-vocab-size "${PADDED_VOCAB_SIZE}"
  --attention-backend flash
  "${FP4_ARGS[@]}"
  "${QAT_BF16_LAYER_ARGS[@]}"
)

EXE="${MEGATRON_ROOT}/pretrain_gpt.py"
NSYS_OUTPUT="${LOG_DIR}/nsys/profile_rank"
TRAIN_LOG="${LOG_DIR}/train.log"

CMD=(
  torchrun "${DISTRIBUTED_ARGS[@]}" "${EXE}"
  "${GPT_ARGS[@]}"
  "${DATA_ARGS[@]}"
  "${MOE_ARGS[@]}"
  "${OUTPUT_ARGS[@]}"
  "${OPTIMIZE_ARGS[@]}"
  "${TRAIN_ARGS[@]}"
  "${MODEL_PARALLEL_ARGS[@]}"
  "${PROFILER_ARGS[@]}"
  "${INFIX_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

run_train() {
  if [[ "${PROFILE}" == "nsys" ]]; then
    mkdir -p "$(dirname "${NSYS_OUTPUT}")"
    nsys profile \
      -s none \
      -t nvtx,cuda \
      -o "${NSYS_OUTPUT}" \
      --force-overwrite true \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop \
      "${CMD[@]}"
  else
    "${CMD[@]}"
  fi
}

# Capture stdout/stderr for throughput parsing.
set +e
run_train 2>&1 | tee "${TRAIN_LOG}"
rc=${PIPESTATUS[0]}
set -e

export TRAIN_LOG TRAIN_RC="${rc}" BACKEND_NAME="${BACKEND}" LOG_DIR

# Lightweight summary: last throughput / elapsed lines from Megatron log.
python3 - <<'PY' || true
from pathlib import Path
import json, os, re

log = Path(os.environ["TRAIN_LOG"])
text = log.read_text(errors="ignore") if log.exists() else ""
tps = [float(x) for x in re.findall(r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)", text)]
elapsed = [float(x) for x in re.findall(r"elapsed time per iteration \(ms\):\s*([0-9.]+)", text)]
if not elapsed:
    elapsed = [float(x) for x in re.findall(r"iteration\s+\d+/.*?([0-9.]+)\s*ms", text)]
summary = {
    "backend": os.environ.get("BACKEND_NAME", ""),
    "log": str(log),
    "exit_code": int(os.environ.get("TRAIN_RC", "0")),
    "throughput_tflops_gpu_samples": tps[-5:] if tps else [],
    "elapsed_ms_per_iter_samples": elapsed[-10:] if elapsed else [],
    "throughput_tflops_gpu_median_last5": (
        sorted(tps[-5:])[len(tps[-5:]) // 2] if tps else None
    ),
    "elapsed_ms_median_last10": (
        sorted(elapsed[-10:])[len(elapsed[-10:]) // 2] if elapsed else None
    ),
}
out = Path(os.environ["LOG_DIR"]) / "perf_summary.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print("wrote", out)
print(json.dumps(summary, indent=2))
PY

exit "${rc}"
