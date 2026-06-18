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
export CUDA_VISIBLE_DEVICES=6,7
export QAT_PARAMS=5
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
mkdir -p $TRITON_CACHE_DIR $NCCL_TMPDIR


TP=2
PP=1
CP=1

GPUS_PER_NODE=2
EXP_NAME="${SLURM_JOB_NAME}_$(basename "${BASH_SOURCE[0]}" .sh)_N${NNODES}_g${GPUS_PER_NODE}_tp${TP}_pp${PP}_cp${CP}_fp4"
echo "EXP_NAME: $EXP_NAME"
LOG_DIR="/home/admin/log/exp/${EXP_NAME}"
TENSORBOARD_DIR="$LOG_DIR/tensorboard"
DATA_PATH=""
DATA_CACHE_PATH=""
TOKENIZER_PATH=""

SEQ_LENGTH=4096
TRAIN_ITERS=125000
EVAL_ITERS=50
SAVE_ITERS=250

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
    --lr-warmup-iters 2000
    --attention-dropout 0.0
    --init-method-std 0.01
    --hidden-dropout 0.0
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
    --initial-loss-scale 4096
    --bf16
    --train-iters ${TRAIN_ITERS}
    --seq-length ${SEQ_LENGTH}
    # --lr-decay-style WSD
    # --lr-warmup-iters $WARMUP_STEPS
    # --lr-wsd-decay-style cosine
    # --lr-wsd-decay-iters $WSD_DECAY_STEPS
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --context-parallel-size ${CP}
    --micro-batch-size 1
    --global-batch-size 128
    --sequence-parallel
    --overlap-grad-reduce
    --overlap-param-gather
    --tp-comm-overlap
)

GPT_ARGS=(
    --qk-layernorm
    --max-position-embeddings ${SEQ_LENGTH}
    --num-layers 36
    --hidden-size 2560
    --ffn-hidden-size 9728
    --num-attention-heads 32
    --kv-channels 128
    --rotary-base 1000000
    --disable-bias-linear
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --attention-softmax-in-fp32
    --group-query-attention
    --num-query-groups 8
    --norm-epsilon 1e-6
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
    --eval-interval ${EVAL_ITERS}
    --save ${LOG_DIR}/checkpoints
    --load ${LOG_DIR}/checkpoints
    --eval-iters 10
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
    --use-pytorch-profiler
    --record-memory-history
)

INFIX_ARGS=(
    # --use-cpu-initialization
    --padded-vocab-size 152064
    --attention-backend flash
    --fp4-format e2m1
    --fp4-recipe nvfp4
    # --fp8-format e4m3
    # --fp8-recipe blockwise
    # --fp8-param-gather
    # --use-precision-aware-optimizer
    # --main-grads-dtype fp32
    # --main-params-dtype fp32
    # --exp-avg-dtype bf16
    # --exp-avg-sq-dtype bf16
)

EXE=${CODE_DIR}/pretrain_gpt.py

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
