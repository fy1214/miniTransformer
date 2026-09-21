#!/bin/bash
# Expert FC1+FC2 comparison runner: sglang vs Megatron (single GPU, TP=EP=PP=1).
#
# One shared logpdiff run directory is created; both collectors write into it
# (sglang/ and megatron/ subdirs), then the run is compared and reported:
#
#   $RUN_DIR/{config.resolved.json, manifest.json, inputs.jsonl,
#             sglang/{index.jsonl,tensors/}, megatron/{index.jsonl,tensors/},
#             metrics.json, metrics.csv, report.html}
#
# Usage:
#   ./run_expert_fc1_compare.sh [input_file] [dump_root]
#
# Overrides (positional arg takes precedence over env var):
#   INPUT_FILE  input JSONL (default: $SCRIPT_DIR/test_inputs_long.jsonl)
#   DUMP_ROOT   run output root (default: $SCRIPT_DIR/runs)
#   RUN_ID      run id (default: UTC timestamp + random suffix)
#   RUN_DIR     full run directory (default: $DUMP_ROOT/$RUN_ID)
#   MODEL_PATH  HF model path (REQUIRED: both collectors)
#   REF_LOAD    Megatron torch_dist checkpoint dir (REQUIRED: megatron collector)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_FILE="${1:-${INPUT_FILE:-$SCRIPT_DIR/test_inputs_long.jsonl}}"
DUMP_ROOT="${2:-${DUMP_ROOT:-$SCRIPT_DIR/runs}}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$(od -An -tx1 -N4 /dev/urandom | tr -d ' ')}"
RUN_DIR="${RUN_DIR:-$DUMP_ROOT/$RUN_ID}"
MODEL_PATH="${MODEL_PATH:-}"
REF_LOAD="${REF_LOAD:-}"

echo "============================================="
echo " Expert FC1 Comparison: sglang vs Megatron"
echo " (one MoE layer, logpdiff run directory)"
echo "============================================="
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "INPUT_FILE: $INPUT_FILE"
echo "RUN_DIR:    $RUN_DIR"

echo ""
echo "=== Checking scripts and inputs ==="
for f in dump_expert_fc1_and_fc2_sglang.py dump_expert_fc1_and_fc2_megatron.py \
         logpdiff/__main__.py logpdiff/cli.py logpdiff/compare.py; do
    if [ ! -f "$SCRIPT_DIR/$f" ]; then
        echo "ERROR: required file missing: $SCRIPT_DIR/$f" >&2
        exit 1
    fi
    echo "$f: OK"
done
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: input file not found: $INPUT_FILE" >&2
    echo "Provide it via positional arg:   $0 /path/to/inputs.jsonl [dump_root]" >&2
    echo "or via environment variable:     INPUT_FILE=/path/to/inputs.jsonl $0" >&2
    exit 1
fi
echo "$INPUT_FILE: OK"
if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH is not set (HF model path, required by both collectors)." >&2
    echo "Run with:  MODEL_PATH=/path/to/hf/model REF_LOAD=/path/to/torch_dist $0" >&2
    exit 1
fi
if [ -z "$REF_LOAD" ]; then
    echo "ERROR: REF_LOAD is not set (Megatron torch_dist checkpoint dir)." >&2
    echo "Run with:  MODEL_PATH=/path/to/hf/model REF_LOAD=/path/to/torch_dist $0" >&2
    exit 1
fi

# Step 1: sglang (HF model) NVFP4 forward, captured into the shared run dir
echo ""
echo "[Step 1/4] sglang/HF NVFP4 forward (collect into $RUN_DIR/sglang/)..."
mkdir -p "$RUN_DIR"
SHOULD_REPLACE_TE_GROUPLINEAR=1 SGLANG_NVFP4_PERTOKEN_SCALE=1 QAT_PARAMS=8 \
NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 \
CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/dump_expert_fc1_and_fc2_sglang.py" \
    --run-dir "$RUN_DIR" \
    --model-path "$MODEL_PATH" \
    --input-file "$INPUT_FILE" \
    --layer-idx 0

# Step 2: Megatron NVFP4 forward, joining the same run dir
echo ""
echo "[Step 2/4] Megatron NVFP4 forward (collect into $RUN_DIR/megatron/)..."
SHOULD_REPLACE_TE_GROUPLINEAR=1 QAT_PARAMS=8 \
NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 \
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 "$SCRIPT_DIR/dump_expert_fc1_and_fc2_megatron.py" \
    --run-dir "$RUN_DIR" \
    --model-path "$MODEL_PATH" \
    --ref-load "$REF_LOAD" \
    --input-file "$INPUT_FILE" \
    --ep-size 1 \
    --layer-idx 0

# Step 3: Compare the run directory (writes metrics.json + metrics.csv)
echo ""
echo "============================================="
echo " COMPARISON (Expert FC1)"
echo "============================================="
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -m logpdiff compare --run-dir "$RUN_DIR"

# Step 4: Render the HTML report from metrics.json
echo ""
echo "[Step 4/4] Rendering report..."
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    python -m logpdiff report --run-dir "$RUN_DIR" --plots

echo ""
echo "Report: $RUN_DIR/report.html"
echo "=== ALL_DONE ==="
