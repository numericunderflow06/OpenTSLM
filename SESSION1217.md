# Session Summary - December 17, 2025

## What Was Done

### 1. Investigated Previous Training Failure
- Merged 5-dataset training had failed with tensor shape mismatch error
- Error: `RuntimeError: stack expects each tensor to be equal size, but got [1, 1000] at entry 0 and [12, 1000] at entry 1`
- Root cause: Different datasets have different channel counts (ECG=12, HAR=3, others=1)

### 2. Verified Fix (Commit `083561b`)
- Fix in `src/time_series_datasets/util.py` - collate function now handles variable channel counts
- Batch returned as list of dicts (not stacked tensor), each sample keeps its own shape

### 3. Ran Sanity Checks
- **Training sanity check**: `sanity_check_merged_training.py` - PASSED
- **Evaluation sanity check**: Custom script testing generation, file I/O, different dataset types - PASSED

### 4. Started Full Merged Training
- Running in persistent tmux session
- GPU 3 (CUDA_VISIBLE_DEVICES=3)
- Model: OpenTSLMFlamingo with Llama-3.2-1B

---

## Where to Find Results

### Training Session
```bash
# Attach to running training
tmux attach -t merged_training

# Detach without stopping: Ctrl+B, then D

# Check if still running
tmux list-sessions

# View live log
tail -f /local/home/wangni/OpenTSLM/merged_training_run.log
```

### Results Directory
```
results_merged/Llama_3_2_1B/OpenTSLMFlamingo/
├── checkpoints/
│   ├── best_model.pt              # Best model weights
│   └── loss_history.txt           # Per-epoch train/val losses
└── results/
    ├── merged_test_predictions.jsonl
    ├── merged_test_metrics.json
    ├── timeseriesexam1_predictions.jsonl
    ├── timeseriesexam1_metrics.json   # <-- KEY METRIC: accuracy
    └── final_results.json             # Combined summary
```

### Key Metric to Check
```bash
cat results_merged/Llama_3_2_1B/OpenTSLMFlamingo/results/timeseriesexam1_metrics.json
```
Look for `"accuracy"` - this is the TimeSeriesExam1 benchmark score.

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | OpenTSLMFlamingo |
| LLM | meta-llama/Llama-3.2-1B |
| GPU | 3 |
| Epochs | 30 (max, with early stopping) |
| Batch size | 4 |
| Early stopping | 5 epochs without improvement |

### Datasets Merged
| Dataset | Train | Val | Test | Channels |
|---------|-------|-----|------|----------|
| TSQA | 38,400 | 4,800 | 4,800 | 1 |
| M4 | 80,000 | 10,000 | 10,000 | 1 |
| HAR CoT | 68,542 | 8,718 | 8,222 | 3 |
| SleepEDF CoT | ~TBD | ~TBD | ~TBD | 1 |
| ECG-QA CoT | ~TBD | ~TBD | ~TBD | 12 |

---

## Commands Reference

```bash
# Kill training if needed
tmux kill-session -t merged_training

# Re-run training manually
cd /local/home/wangni/OpenTSLM
CUDA_VISIBLE_DEVICES=3 python merged_training.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --gradient_checkpointing

# Run evaluation only (after training completes)
CUDA_VISIBLE_DEVICES=3 python merged_training.py \
    --model OpenTSLMFlamingo \
    --llm_id meta-llama/Llama-3.2-1B \
    --eval_only

# Quick sanity check
CUDA_VISIBLE_DEVICES=3 python sanity_check_merged_training.py \
    --samples_per_dataset 10 \
    --batch_size 4 \
    --num_training_steps 3
```

---

## Model Architecture Summary

The model uses **Flamingo-style cross-attention**:
1. Time series values encoded via TransformerCNN Encoder
2. Projected to LLM embedding space via MLP Projector
3. LLM attends to time series embeddings via cross-attention layers
4. Text prompts provide context (pre_prompt + post_prompt)
5. Model generates answer autoregressively

Loss computed only on answer tokens during training.
