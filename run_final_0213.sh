#!/bin/bash
# Final experiments: eval scratch_zuco1, then train+eval scratch_zuco2
set -e

export CUDA_VISIBLE_DEVICES=1
LLM_ID="meta-llama/Llama-3.2-3B"
MODEL="OpenTSLMFlamingo"
BATCH_SIZE=2
LOGDIR="logs/zuco_et_experiments"

# --- Eval scratch_zuco1 (killed at epoch 4, best at epoch 3) ---
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

# --- Train + Eval scratch_zuco2 ---
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
