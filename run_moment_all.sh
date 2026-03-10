#!/bin/bash
# Run MOMENT encoder experiments on all remaining datasets (subsampled)
# GPUs: 0, 1, 3 (avoiding GPU 2)
# Already completed: stage4_sleep_cot
set -e

LLM_ID="meta-llama/Llama-3.2-3B"
MODEL="OpenTSLMSP"
BATCH_SIZE=2
ENCODER="moment"
MAX_SAMPLES=500
LOGDIR="logs/moment_experiments"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "  MOMENT Encoder — Multi-Dataset Runs"
echo "  LLM: $LLM_ID"
echo "  Encoder: $ENCODER"
echo "  Batch size: $BATCH_SIZE"
echo "  Max samples per split: $MAX_SAMPLES"
echo "=========================================="

# GPU 0: stage1_mcq (TSQA), then stage9b_et2_reading_task (ZuCo 2.0 ET) as separate processes
(
    CUDA_VISIBLE_DEVICES=0 python curriculum_learning.py \
        --model $MODEL \
        --llm_id "$LLM_ID" \
        --stages stage1_mcq \
        --encoder_type $ENCODER \
        --experiment_tag moment \
        --batch_size $BATCH_SIZE \
        --max_samples $MAX_SAMPLES \
        2>&1 | tee "$LOGDIR/moment_gpu0_mcq.log"

    CUDA_VISIBLE_DEVICES=0 python curriculum_learning.py \
        --model $MODEL \
        --llm_id "$LLM_ID" \
        --stages stage9b_et2_reading_task \
        --encoder_type $ENCODER \
        --experiment_tag moment \
        --batch_size $BATCH_SIZE \
        --max_samples $MAX_SAMPLES \
        2>&1 | tee "$LOGDIR/moment_gpu0_et2.log"
) &
PID_GPU0=$!

# GPU 1: stage6_financial_reports
CUDA_VISIBLE_DEVICES=1 python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage6_financial_reports \
    --encoder_type $ENCODER \
    --experiment_tag moment \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    2>&1 | tee "$LOGDIR/moment_gpu1_financial.log" &
PID_GPU1=$!

# GPU 3: stage9_et_reading_task (ZuCo 1.0 ET)
CUDA_VISIBLE_DEVICES=3 python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage9_et_reading_task \
    --encoder_type $ENCODER \
    --experiment_tag moment \
    --batch_size $BATCH_SIZE \
    --max_samples $MAX_SAMPLES \
    2>&1 | tee "$LOGDIR/moment_gpu3_et1.log" &
PID_GPU3=$!

echo ""
echo "Launched 3 GPU jobs:"
echo "  GPU 0 (PID $PID_GPU0): stage1_mcq → stage9b_et2_reading_task (sequential)"
echo "  GPU 1 (PID $PID_GPU1): stage6_financial_reports"
echo "  GPU 3 (PID $PID_GPU3): stage9_et_reading_task"
echo ""
echo "Monitor with: tail -f $LOGDIR/moment_gpu*.log"

wait $PID_GPU0 $PID_GPU1 $PID_GPU3
echo ""
echo "All MOMENT experiments finished!"
