#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE8_CKPT="results/Llama_3_2_3B/OpenTSLMFlamingo/stage8_eeg_sentiment/checkpoints/best_model.pt"

python curriculum_learning.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-3B \
    --stages stage9b_et2_reading_task \
    --eval_only \
    --init_checkpoint "$STAGE8_CKPT" \
    --experiment_tag fullcurr_zuco2 \
    --batch_size 2 \
    --gradient_checkpointing \
    2>&1 | tee logs/zuco_et_experiments/fullcurr_zuco2_eval.log
