#!/bin/bash
# Remaining ZuCo experiments on GPU 1 — 2026-02-13
set -e

export CUDA_VISIBLE_DEVICES=1
LLM_ID="meta-llama/Llama-3.2-3B"
MODEL="OpenTSLMFlamingo"
BATCH_SIZE=2
LOGDIR="logs/zuco_et_experiments"

SLEEP_CHECKPOINT="/home/wangni/.cache/huggingface/hub/models--OpenTSLM--llama-3.2-3b-sleep-flamingo/snapshots/2afbbbbd6e2006f8f5d70a21b03c60af3f050615/model_checkpoint.pt"

# --- Experiment 1: Eval-only for zuco1_sleep (killed during training, best checkpoint saved) ---
echo "=========================================="
echo "  Eval: Sleep → ZuCo 1.0"
echo "=========================================="
python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage9_et_reading_task \
    --eval_only \
    --init_checkpoint "$SLEEP_CHECKPOINT" \
    --experiment_tag zuco1_sleep \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    2>&1 | tee "$LOGDIR/zuco1_sleep_eval.log"

# --- Experiment 2: From scratch → ZuCo 1.0 ---
echo ""
echo "=========================================="
echo "  Experiment: From scratch → ZuCo 1.0"
echo "=========================================="
python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage9_et_reading_task \
    --experiment_tag scratch_zuco1 \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    2>&1 | tee "$LOGDIR/scratch_zuco1.log"

# --- Experiment 3: From scratch → ZuCo 2.0 ---
echo ""
echo "=========================================="
echo "  Experiment: From scratch → ZuCo 2.0"
echo "=========================================="
python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage9b_et2_reading_task \
    --experiment_tag scratch_zuco2 \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    2>&1 | tee "$LOGDIR/scratch_zuco2.log"

echo ""
echo "All experiments finished!"
