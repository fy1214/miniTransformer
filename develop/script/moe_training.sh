#!/bin/bash

NNODES=1
RDZV_ENDPOINT="localhost:29400"

# 创建一个数组存储其他参数
ARGS=()

# 解析所有参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --rdzv_endpoint)
            RDZV_ENDPOINT="$2"
            shift 2
            ;;
        *)
            # 将其他参数添加到数组中
            ARGS+=("$1")
            shift
            ;;
    esac
done

CODE_DIR=/root/Megatron-LM
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=$PYTHONPATH:$CODE_DIR
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
export TRITON_CACHE_DIR=/tmp/$USER/triton
export NCCL_TMPDIR=/tmp/$USER/nccl
# export NCCL_SOCKET_IFNAME=bond0

export NVTE_NVFP4_DISABLE_RHT=0
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=1

mkdir -p $TRITON_CACHE_DIR $NCCL_TMPDIR


TP=1
PP=1
CP=1
EP=4

GPUS_PER_NODE=8
EXP_NAME="${SLURM_JOB_NAME}_$(basename "${BASH_SOURCE[0]}" .sh)_N${NNODES}_g${GPUS_PER_NODE}_tp${TP}_pp${PP}_cp${CP}_ep${EP}_fp4_te"
echo "EXP_NAME: $EXP_NAME"
LOG_DIR="/home/admin/log/exp/${EXP_NAME}"
TENSORBOARD_DIR="$LOG_DIR/tensorboard"
DATA_PATH=""
DATA_CACHE_PATH=""
TOKENIZER_PATH=""

SEQ_LENGTH=4096
TRAIN_ITERS=30
EVAL_ITERS=0
SAVE_ITERS=10000

DISTRIBUTED_ARGS=(
    --nnodes=$NNODES
    --nproc_per_node=$GPUS_PER_NODE
    --rdzv_endpoint=$RDZV_ENDPOINT
    --rdzv_backend=c10d
)

OPTIMIZE_ARGS=(
    --use-flash-attn
    --use-rotary-position-embeddings
    --no-masked-softmax-fusion
    --use-distributed-optimizer
    --bf16
    --accumulate-allreduce-grads-in-fp32
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
    --train-iters ${TRAIN_ITERS}
    --seq-length ${SEQ_LENGTH}
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    --expert-model-parallel-size ${EP}
    --micro-batch-size 1
    --global-batch-size 128
    --overlap-grad-reduce
    --overlap-param-gather
)

# Qwen3-30B-A3B: hidden=2048, layers=48, heads=32, kv_heads=4, head_dim=128
GPT_ARGS=(
    --qk-layernorm
    --max-position-embeddings ${SEQ_LENGTH}
    --num-layers 48
    --hidden-size 2048
    --ffn-hidden-size 768
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

# MoE: 128 experts, top-8, per-expert ffn=768
MOE_ARGS=(
    --num-experts 128
    --moe-router-topk 8
    --moe-ffn-hidden-size 768
    --moe-grouped-gemm
    --moe-aux-loss-coeff 0.001
    --moe-token-dispatcher-type alltoall
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --data-cache-path ${DATA_CACHE_PATH}
    --tokenizer-model ${TOKENIZER_PATH}
    --tokenizer-type HuggingFaceTokenizer
    --split 90,5,5
    --no-create-attention-mask-in-dataloader
    --num-workers 6
)

OUTPUT_ARGS=(
    --log-interval 1
    --save-interval ${SAVE_ITERS}
    --eval-interval 10000
    --save ${LOG_DIR}/checkpoints
    --load ${LOG_DIR}/checkpoints
    --eval-iters 0
    --no-load-optim
    --no-load-rng
    --log-throughput
    --log-timers-to-tensorboard
    --log-params-norm
    --log-num-zeros-in-grad
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --tensorboard-dir ${TENSORBOARD_DIR}
    --wandb-project pretrain_v1
    --wandb-exp-name ${EXP_NAME}
    --wandb-save-dir ${LOG_DIR}
)

PROFILER_ARGS=(
    --profile
    --profile-step-start 20
    --profile-step-end 24
    --profile-ranks 0
)

INFIX_ARGS=(
    --padded-vocab-size 151936
    --attention-backend flash
    --fp4-format e2m1
    --fp4-recipe nvfp4
)

EXE=${CODE_DIR}/pretrain_gpt.py
NSYS_OUTPUT="${LOG_DIR}/nsys/profile_rank"

mkdir -p "$(dirname "$NSYS_OUTPUT")"

nsys profile \
    -s none \
    -t nvtx,cuda \
    -o "${NSYS_OUTPUT}" \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    torchrun "${DISTRIBUTED_ARGS[@]}" "${EXE}" \
    "${GPT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${OPTIMIZE_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${PROFILER_ARGS[@]}" \
    "${INFIX_ARGS[@]}"