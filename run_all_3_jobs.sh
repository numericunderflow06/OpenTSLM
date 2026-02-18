#!/bin/bash
# Run all 3 ZuCo ET experiments sequentially on GPU 1
# Job 1 is already running - this script handles Job 2 and Job 3 after Job 1 finishes.

set -e

HAR_CKPT="/home/wangni/.cache/huggingface/hub/models--OpenTSLM--llama-3.2-3b-har-flamingo/snapshots/d7b9fb9f3fe1508ed3fdaa95c000c58a0ed78e75/model_checkpoint.pt"
SLEEP_CKPT="/home/wangni/.cache/huggingface/hub/models--OpenTSLM--llama-3.2-3b-sleep-flamingo/snapshots/2afbbbbd6e2006f8f5d70a21b03c60af3f050615/model_checkpoint.pt"
LLM_ID="meta-llama/Llama-3.2-3B"
LOGDIR="logs/zuco_et_experiments"

echo "[$(date)] Starting Job 2: ZuCo 2.0 + sleep-flamingo"
CUDA_VISIBLE_DEVICES=1 python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id "$LLM_ID" \
    --stages stage9b_et2_reading_task \
    --init_checkpoint "$SLEEP_CKPT" \
    --experiment_tag zuco2_sleep \
    --batch_size 2 \
    --gradient_checkpointing \
    > "$LOGDIR/zuco2_sleep.log" 2>&1
echo "[$(date)] Job 2 finished"

echo "[$(date)] Starting Job 3: ZuCo 1.0 + HAR-flamingo"
CUDA_VISIBLE_DEVICES=1 python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id "$LLM_ID" \
    --stages stage9_et_reading_task \
    --init_checkpoint "$HAR_CKPT" \
    --experiment_tag zuco1_har \
    --batch_size 2 \
    --gradient_checkpointing \
    > "$LOGDIR/zuco1_har.log" 2>&1
echo "[$(date)] Job 3 finished"

echo "[$(date)] All 3 jobs complete!"
