#!/bin/bash
#
# Launch 3 ZuCo eye-tracking experiments for CoNLL 2026 paper.
#
# Job 1: ZuCo 2.0 ET + HAR-flamingo   -> zuco2_har
# Job 2: ZuCo 2.0 ET + sleep-flamingo -> zuco2_sleep
# Job 3: ZuCo 1.0 ET + HAR-flamingo   -> zuco1_har
#
# Prerequisites:
#   1. Download ZuCo 2.0 TSR Matlab files: bash download_zuco2_tsr.sh
#   2. Set checkpoint paths below to actual .pt files
#

set -e

# ======================== CONFIGURATION ========================
# Set these to the actual checkpoint paths before running
HAR_CHECKPOINT="${HAR_CHECKPOINT:-results/Llama_3_2_3B/OpenTSLMFlamingo/stage3_cot/checkpoints/best_model.pt}"
SLEEP_CHECKPOINT="${SLEEP_CHECKPOINT:-results/Llama_3_2_3B/OpenTSLMFlamingo/stage4_sleep_cot/checkpoints/best_model.pt}"

LLM_ID="meta-llama/Llama-3.2-3B"
MODEL="OpenTSLMFlamingo"
BATCH_SIZE=2

# GPU assignment (adjust for your setup)
GPU_JOB1=0
GPU_JOB2=1
GPU_JOB3=2
# ===============================================================

LOGDIR="logs/zuco_et_experiments"
mkdir -p "$LOGDIR"

echo "========================================"
echo "ZuCo Eye-Tracking Experiments Launcher"
echo "========================================"
echo "HAR checkpoint:   $HAR_CHECKPOINT"
echo "Sleep checkpoint: $SLEEP_CHECKPOINT"
echo "LLM ID:           $LLM_ID"
echo "Log directory:    $LOGDIR"
echo ""

# Verify checkpoints exist
for ckpt in "$HAR_CHECKPOINT" "$SLEEP_CHECKPOINT"; do
    if [ ! -f "$ckpt" ]; then
        echo "WARNING: Checkpoint not found: $ckpt"
        echo "  Set HAR_CHECKPOINT / SLEEP_CHECKPOINT env vars or edit this script."
    fi
done

# --- Job 1: ZuCo 2.0 ET + HAR-flamingo ---
echo "[Job 1] ZuCo 2.0 ET + HAR-flamingo (GPU $GPU_JOB1) -> zuco2_har"
CUDA_VISIBLE_DEVICES=$GPU_JOB1 python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage9b_et2_reading_task \
    --init_checkpoint "$HAR_CHECKPOINT" \
    --experiment_tag zuco2_har \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    > "$LOGDIR/zuco2_har.log" 2>&1 &
PID1=$!
echo "  PID: $PID1, Log: $LOGDIR/zuco2_har.log"

# --- Job 2: ZuCo 2.0 ET + sleep-flamingo ---
echo "[Job 2] ZuCo 2.0 ET + sleep-flamingo (GPU $GPU_JOB2) -> zuco2_sleep"
CUDA_VISIBLE_DEVICES=$GPU_JOB2 python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage9b_et2_reading_task \
    --init_checkpoint "$SLEEP_CHECKPOINT" \
    --experiment_tag zuco2_sleep \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    > "$LOGDIR/zuco2_sleep.log" 2>&1 &
PID2=$!
echo "  PID: $PID2, Log: $LOGDIR/zuco2_sleep.log"

# --- Job 3: ZuCo 1.0 ET + HAR-flamingo ---
echo "[Job 3] ZuCo 1.0 ET + HAR-flamingo (GPU $GPU_JOB3) -> zuco1_har"
CUDA_VISIBLE_DEVICES=$GPU_JOB3 python curriculum_learning.py \
    --model $MODEL \
    --llm_id "$LLM_ID" \
    --stages stage9_et_reading_task \
    --init_checkpoint "$HAR_CHECKPOINT" \
    --experiment_tag zuco1_har \
    --batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    > "$LOGDIR/zuco1_har.log" 2>&1 &
PID3=$!
echo "  PID: $PID3, Log: $LOGDIR/zuco1_har.log"

echo ""
echo "All 3 jobs launched. PIDs: $PID1, $PID2, $PID3"
echo ""
echo "Monitor with:"
echo "  tail -f $LOGDIR/zuco2_har.log"
echo "  tail -f $LOGDIR/zuco2_sleep.log"
echo "  tail -f $LOGDIR/zuco1_har.log"
echo ""
echo "Results will be saved to:"
echo "  results/Llama_3_2_3B/OpenTSLMFlamingo_zuco2_har/stage9b_et2_reading_task/"
echo "  results/Llama_3_2_3B/OpenTSLMFlamingo_zuco2_sleep/stage9b_et2_reading_task/"
echo "  results/Llama_3_2_3B/OpenTSLMFlamingo_zuco1_har/stage9_et_reading_task/"

wait
echo "All jobs finished."
