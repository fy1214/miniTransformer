#!/bin/bash
set -e
BENCH=/home/lmsys/mingfa/benchmark
MEGATRON=/opt/Megatron-Bridge/3rdparty/Megatron-LM
mkdir -p ${BENCH}/results

# 8 GPU: TP=2 DP=2, seq=2048 MBS=8, M=32768, GBS=512
COMMON_ARGS="\
    --num-layers 36 --hidden-size 4096 --ffn-hidden-size 12288 \
    --num-attention-heads 32 --group-query-attention --num-query-groups 8 \
    --kv-channels 128 --qk-layernorm --swiglu \
    --use-rotary-position-embeddings --rotary-base 1000000 \
    --normalization RMSNorm --norm-epsilon 1e-6 \
    --disable-bias-linear --untie-embeddings-and-output-weights \
    --vocab-size 151936 --seq-length 2048 --max-position-embeddings 32768 \
    --tokenizer-type NullTokenizer \
    --mock-data \
    --tensor-model-parallel-size 2 --pipeline-model-parallel-size 1 \
    --sequence-parallel \
    --micro-batch-size 8 --global-batch-size 512 \
    --train-iters 150 --log-interval 10 \
    --eval-interval 150 --eval-iters 0 \
    --save-interval 99999 --no-save-optim --no-save-rng \
    --no-load-optim --no-load-rng \
    --optimizer adam --use-distributed-optimizer \
    --overlap-grad-reduce --overlap-param-gather \
    --lr 1e-4 --min-lr 1e-5 --lr-decay-style cosine \
    --weight-decay 0.01 --clip-grad 1.0 --adam-beta1 0.9 --adam-beta2 0.95 \
    --bf16 \
    --attention-dropout 0 --hidden-dropout 0 \
    --recompute-activations \
    --log-throughput"

run_test() {
    local NAME=$1
    shift
    local EXTRA_ARGS="$@"
    echo ""
    echo "========================================"
    echo "TEST: ${NAME}"
    echo "EXTRA: ${EXTRA_ARGS}"
    echo "========================================"
    rm -f /tmp/ub-ipc-socket-*
    sleep 2
    docker run --rm \
      --gpus all \
      --network=host \
      --shm-size=128g --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      -v ${BENCH}:/workspace/benchmark \
      -e PYTHONPATH=${MEGATRON} \
      -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
      -e CUDA_VISIBLE_DEVICES=4,5,6,7 \
      nvcr.io/nvidia/nemo:26.02 \
      torchrun --nproc_per_node=4 --master_addr=localhost --master_port=12370 \
        ${MEGATRON}/pretrain_gpt.py \
        ${COMMON_ARGS} \
        ${EXTRA_ARGS} \
        2>&1 | tee ${BENCH}/results/4gpu_s2k_mbs8_te_${NAME}.log
    echo "DONE: ${NAME} (exit=$?)"
}

# BF16 + TE attn scope
run_test "bf16" \
    --tp-comm-overlap

# NVFP4 + TE attn scope
run_test "nvfp4" \
    --fp4-format e2m1 --fp4-recipe nvfp4

echo "ALL DONE"