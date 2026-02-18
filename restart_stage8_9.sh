#!/bin/bash
# Restart training from stage 8+9 with fixed sentiment labels
# Run this after stage 7 completes (check training_zuco.log for completion)

# Kill any existing training processes
pkill -f "curriculum_learning.*Llama-3.2-3B" 2>/dev/null
sleep 5

# Launch with all 3 stages - stage 7 will be auto-skipped (completed)
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 nohup python -u curriculum_learning.py \
  --model OpenTSLMFlamingo \
  --llm_id meta-llama/Llama-3.2-3B \
  --stages stage7_eeg_reading_task stage8_eeg_sentiment stage9_et_reading_task \
  --gradient_checkpointing \
  --verbose \
  > training_zuco_stage8_9.log 2>&1 &

echo "Launched with PID $!"
echo "Monitor: tail -f training_zuco_stage8_9.log"
