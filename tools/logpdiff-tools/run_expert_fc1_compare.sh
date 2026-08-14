#!/bin/bash
set -e
cd /root/compare_logprob
INPUT_FILE="test_inputs_long.jsonl"

echo "============================================="
echo " Expert FC1 Comparison: sglang vs Megatron"
echo " (Layer 0 fc1: input / QDQ / output)"
echo "============================================="

echo ""
echo "=== Checking scripts ==="
for f in dump_expert_fc1_and_fc2_sglang.py dump_expert_fc1_and_fc2_megatron.py compare_expert_fc1.py "$INPUT_FILE"; do
    echo "$f: $(test -f "$f" && echo OK || echo MISSING)"
done

# Step 1: sglang (HF model) NVFP4 forward
echo ""
echo "[Step 1] sglang/HF NVFP4 forward (capture expert fc1)..."
mkdir -p dump_expert_fc1_and_fc2_sglang
SHOULD_REPLACE_TE_GROUPLINEAR=1 SGLANG_NVFP4_PERTOKEN_SCALE=1 QAT_PARAMS=8 \
NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 \
CUDA_VISIBLE_DEVICES=0 python dump_expert_fc1_and_fc2_sglang.py \
    --input-file "$INPUT_FILE" \
    --output-dir dump_expert_fc1_and_fc2_sglang/ \
    --layer-idx 0

# Step 2: Megatron NVFP4 forward
echo ""
echo "[Step 2] Megatron NVFP4 forward (capture expert fc1)..."
mkdir -p dump_expert_fc1_and_fc2_megatron
SHOULD_REPLACE_TE_GROUPLINEAR=1 QAT_PARAMS=8 \
NVTE_NVFP4_DISABLE_RHT=0 NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0 NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1 \
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 dump_expert_fc1_and_fc2_megatron.py \
    --input-file "$INPUT_FILE" \
    --output-dir dump_expert_fc1_and_fc2_megatron/ \
    --ep-size 1 \
    --layer-idx 0

# Step 3: Compare
echo ""
echo "============================================="
echo " COMPARISON (Expert FC1)"
echo "============================================="
python compare_expert_fc1_and_fc2.py \
    --sglang-dump dump_expert_fc1_and_fc2_sglang/sglang_expert_fc1.pt \
    --megatron-dump dump_expert_fc1_and_fc2_megatron/megatron_expert_fc1.pt

echo ""
echo "=== ALL_DONE ==="
