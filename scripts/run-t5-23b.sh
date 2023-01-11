#!/bin/bash
set -ex

data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_PATH} \
         --data-impl mmap"

BASE_PATH=$PWD/dataset/
DATA_PATH=${BASE_PATH}/BookCorpusDataset_text_document
DS_CONFIG=ds_config.json

# Hostfile path
HF=$PWD/hostfile 

# Disabling tensor/pipeline parallelism

TP=1
PP=5

# HEADS ~= HIDDEN/128

# Model: Benchmark model
NLAYERS=36
HIDDEN=5120
HEADS=80
KV_CHANNELS=64
FFN_HIDDEN_SIZE=10880
SEQ=512

NODES=40
GPN=8

MICRO_BATCH=5

GRAD_ACC_STEP=170
GLOBAL_BATCH=$(( (${GPN} * ${NODES} / ${PP} / ${TP})  * ${MICRO_BATCH} * ${GRAD_ACC_STEP}))

# Initial power scale for loss
SP=15

# Uncomment/comment one of the following blocks.

# For 1T model, start with microbatch=1, try to get 2 and 4.  If OOM w/ 4, use cpu-offloading

# Set to cpu for offloading to cpu for larger models
#OFFLOAD_DEVICE="cpu"
#CPU_OPTIM=" --cpu-optimizer"

# Set to none and empty string for no cpu offloading
OFFLOAD_DEVICE="none"  
CPU_OPTIM=" "

ZERO_STAGE=1
OUTPUT_DIR=ds_z_off-${OFFLOAD_DEVICE}_stage_${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_mb${MICRO_BATCH}_seq${SEQ}_gb${GLOBAL_BATCH}_nodes${NODES}_pp${PP}_gas${GRAD_ACC_STEP}
#OUTPUT_DIR=baseline_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}
mkdir -p $OUTPUT_DIR

cat <<EOT > $DS_CONFIG
{
  "steps_per_print": 2,
  "train_batch_size": $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "gradient_accumulation_steps": $GRAD_ACC_STEP,
  "gradient_clipping": 1.0,
  "optimizer": {
    "type": "adam",
    "params": {
      "lr": 9.25e-5
    }
  },
  "fp16": {
    "enabled": true,
    "autocast": true,
    "initial_scale_power" : $SP,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "overlap_comm": true,
    "allgather_bucket_size": 10000000000,
    "reduce_scatter": true,
    "allgather_partitions": false
  },

  "sparse_gradients": false,

  "wall_clock_breakdown": false,

  "flops_profiler": {
    "enabled": false,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": "./flops_profiler.log"
  }
}
EOT

export NCCL_DEBUG=WARN 

ds_args=" "
ds_args=" --deepspeed ${ds_args}"
if [[ $PP == 1 ]]; then
	ds_args=" --no-pipeline-parallel ${ds_args}" 
fi
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
#ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

deepspeed --force_multi --num_nodes=$NODES --hostfile $HF --no_ssh_check --master_addr 192.169.16.100 --master_port=12345 pretrain_t5.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --kv-channels $KV_CHANNELS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --encoder-seq-length $SEQ \
    --decoder-seq-length $SEQ \
    --hidden-size $HIDDEN \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --num-attention-heads $HEADS \
    --loss-scale $SP \
    --max-position-embeddings $SEQ \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --distributed-backend nccl \
    --data-impl mmap \
    --train-iters 2 \
    --lr-decay-iters 320000 \
    --lr 9.25e-5 \
    --min-lr 9.25e-6 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 2 \
    --eval-interval 1000 \
    --data-path $DATA_PATH \
    --vocab-file $BASE_PATH/bert-vocab2.txt \
    --merge-file $BASE_PATH/gpt2-merges.txt \
    --vocab-extra-ids 100 \
    --save-interval 1000 \
    --clip-grad 1.0 \
    --weight-decay 1e-2 \
    --init-method-std 0.006 \
    --fp16 \
    --tensorboard-dir $OUTPUT_DIR \
    $CPU_OPTIM $ds_args \
    --exit-interval 5000 --DDP-impl local | tee ${OUTPUT_DIR}/output.log
