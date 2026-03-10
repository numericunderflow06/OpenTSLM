#!/bin/bash
# Train stage4_sleep_cot with pretrained MOMENT encoder on GPU 2
set -e

export CUDA_VISIBLE_DEVICES=2
LLM_ID="meta-llama/Llama-3.2-3B"
MODEL="OpenTSLMSP"
BATCH_SIZE=2
ENCODER="moment"
LOGDIR="logs/moment_experiments"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "  MOMENT Encoder — Stage 4 Sleep CoT"
echo "  LLM: $LLM_ID"
echo "  Encoder: $ENCODER"
echo "  Batch size: $BATCH_SIZE"
echo "=========================================="

python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage4_sleep_cot \
    --encoder_type $ENCODER \
    --experiment_tag moment \
    --batch_size $BATCH_SIZE \
    2>&1 | tee "$LOGDIR/moment_stage4_sleep.log"

echo ""
echo "MOMENT stage4_sleep_cot experiment finished!"
