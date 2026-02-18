#!/bin/bash
# Eval-only runs for scratch experiments with smaller batch size
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LLM_ID="meta-llama/Llama-3.2-3B"
MODEL="OpenTSLMFlamingo"
BATCH_SIZE=1
LOGDIR="logs/zuco_et_experiments"

# --- Eval scratch_zuco1 ---
echo "=========================================="
echo "  Eval: scratch → ZuCo 1.0"
echo "=========================================="
python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage9_et_reading_task \
    --eval_only \
    --experiment_tag scratch_zuco1 \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    2>&1 | tee "$LOGDIR/scratch_zuco1_eval.log"

# --- Train scratch_zuco2 ---
echo ""
echo "=========================================="
echo "  Train: scratch → ZuCo 2.0"
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
echo "All done!"
