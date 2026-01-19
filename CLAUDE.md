# OpenTSLM Polymarket Training - Reference Guide

## ⚠️ CRITICAL Requirements

### 1. File Paths: MUST use `/local/home/wangni/`

**ALL files, downloads, and cached files MUST be in `/local/home/wangni/` - NOT `/home/wangni/`**

### 2. Conda Environment: MUST use `tslm`

**ALL Python scripts MUST be run in the `tslm` conda environment - NOT base or other environments**

```bash
# Always use one of these:
conda activate tslm
# OR
conda run -n tslm python your_script.py
```

### Common Mistake: HuggingFace Cache

HuggingFace and other libraries often cache files to `~/.cache/` which resolves to `/home/wangni/.cache/` by default. This is WRONG for our setup.

**Always set these environment variables before running ANY Python scripts:**

```bash
export HF_HOME=/local/home/wangni/.cache/huggingface
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TRANSFORMERS_CACHE=/local/home/wangni/.cache/huggingface/transformers
```

**Check your cache locations:**
```bash
# WRONG - will cache to /home/wangni
ls ~/.cache/huggingface

# CORRECT - should be here
ls /local/home/wangni/.cache/huggingface
```

### Critical Checklist Before Running Any Script
- ✅ Using `tslm` conda environment (`conda activate tslm` or `conda run -n tslm`)
- ✅ Set `HF_HOME=/local/home/wangni/.cache/huggingface`
- ✅ Set `HF_TOKEN="YOUR_HF_TOKEN_HERE"`
- ✅ Set `TRANSFORMERS_CACHE=/local/home/wangni/.cache/huggingface/transformers`
- ✅ Data in: `/local/home/wangni/polymarket_data/`
- ✅ Results in: `/local/home/wangni/results/`
- ✅ Cache in: `/local/home/wangni/.cache/`
- ❌ NEVER: `/home/wangni/` (anywhere!)
- ❌ NEVER: base or other conda environments

## Dataset Structure

### Overview
- **Total samples**: 1,774 (887 markets × 2 question types)
- **Data source**: 12 months of Polymarket data (962 markets downloaded)
- **Location**: `/local/home/wangni/polymarket_data/`

### Time Series Splitting

Each time series (typically 278 values) is **split in half**:

```
Full series: [v1, v2, ..., v139, v140, v141, ..., v278]
                          ↓ Split at midpoint ↓
First half:  [v1, v2, ..., v139]          ← Model sees this
Second half:             [v140, ..., v278] ← Ground truth computation ONLY
```

**Critical**: The model NEVER sees the second half during training or inference!

### Two Question Types per Market

1. **Past Trend Question** (887 samples)
   - Input: First half of time series
   - Question: "What was the trend in the first period?"
   - Ground truth: Computed from first half's slope via linear regression

2. **Future Forecast Question** (887 samples)
   - Input: First half of time series (same as above!)
   - Question: "What is the trend in the second period?"
   - Ground truth: Computed from second half's slope (model must predict!)

### Ground Truth Computation
- Linear regression on each half
- If slope > 0: "increasing"
- If slope < 0: "decreasing"
- If slope = 0: Sample excluded

### Label Distribution
- Increasing: 36.4%
- Decreasing: 63.6%

## Question Format

### Actual Prompt Sent to LLM

```
This is the prediction market probability time series:

<35 embedding tokens from encoder - NOT raw text!>

Based on the time series data above, what was the trend in the first period?

Answer with ONLY one word: increasing or decreasing
```

Or for future forecast:

```
This is the prediction market probability time series:

<35 embedding tokens from encoder>

Based on the time series data above, what is the trend in the second period?

Answer with ONLY one word: increasing or decreasing
```

### Key Points
- Time series is **embedded**, not converted to text tokens
- Explicit instruction: "Answer with ONLY one word"
- Fuzzy parsing: Extract "increasing" or "decreasing" from any output

### ⚠️ Important: Parsing Logic Must Match Original OpenTSLM

**The current fuzzy parsing implementation may not match the original OpenTSLM parsing logic.**

Before training extensively, verify that our answer extraction logic matches what the original OpenTSLM code uses. If there's a mismatch:
1. Check the original OpenTSLM repository for their parsing implementation
2. Update our `extract_answer()` function in `train_model.py` and `test_untrained_encoder.py`
3. Re-run evaluation to ensure consistency

**Current implementation** (may need updating):
```python
def extract_answer(text: str) -> str:
    # Looks for 'increasing' or 'decreasing' anywhere in output
    # Returns first occurrence if both found
```

**TODO**: Compare with original OpenTSLM and update if needed.

## OpenTSLM Embedding Pipeline

### The Magic: 139 Values → 35 Tokens

```
Step 1: Raw time series
  [0.25, 0.25, 0.25, ..., 0.0035]  (139 float values)

Step 2: Pad to multiple of patch_size (4)
  [0.25, 0.25, ..., 0.0035, 0.0]   (140 values)

Step 3: Encoder (TransformerCNNEncoder)
  - Patchify: 140 ÷ 4 = 35 patches
  - CNN feature extraction
  - Transformer processing
  Output: [1, 35, encoder_dim]

Step 4: Projector (MLPProjector)
  - Project to LLM hidden size
  Output: [1, 35, hidden_size]

Result: 35 learned embeddings representing the time series!
```

### Comparison to Baseline (Text Encoding)

| Approach | Input | Output Tokens |
|----------|-------|---------------|
| **OpenTSLM** | 139 float values | **35 tokens** |
| **Baseline (text)** | "0.25, 0.25, ..." | **~600-800 tokens** |

### How Embeddings are Interleaved

The final sequence fed to the LLM:

```
[
  <"This" embed>,
  <"is" embed>,
  <"the" embed>,
  ...
  <TS patch 1 embed>,  ← Learned from encoder!
  <TS patch 2 embed>,  ← Learned from encoder!
  ...
  <TS patch 35 embed>, ← Learned from encoder!
  <"Based" embed>,
  <"on" embed>,
  ...
]
```

Total tokens per sample: ~72 (12 text + 35 TS + 25 text)

### Code Reference
- Embedding pipeline: `src/model/llm/OpenTSLMSP.py` (lines 181-319)
- Encoder: `src/model/encoder/TransformerCNNEncoder.py`
- Projector: `src/model/projector/MLPProjector.py`

## Model Architecture

### LLM (Frozen)
- **Purpose**: Generate trend predictions based on time series embeddings
- **Status**: Frozen (no training on LLM weights)
- **Requirements**: Model must be cached in `/local/home/wangni/.cache/huggingface/`
- **Examples**: Can use Gemma-3-270m, Qwen3-14B, or other models

### Trainable Components
1. **Encoder**: TransformerCNNEncoder
   - Converts time series patches to embeddings
   - Learns to extract meaningful patterns from time series data

2. **Projector**: MLPProjector
   - Projects encoder output to LLM hidden dimension
   - Bridges the gap between encoder and LLM spaces

### Training Configuration

Typical training setup (adjust as needed):
```python
BATCH_SIZE = 4  # Per GPU
NUM_EPOCHS = 20
LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_FRAC = 0.03
GRAD_CLIP_NORM = 1.0
EVAL_EVERY = 100 steps
SAVE_EVERY = 500 steps
```

### Checkpoints
- **Best model**: `/local/home/wangni/results/polymarket_checkpoints/best_model.pt`
- **Periodic**: `/local/home/wangni/results/polymarket_checkpoints/checkpoint_step_{N}.pt`

## ⚠️ CRITICAL: Logging Requirements

**Logging is EXTREMELY IMPORTANT for this project.**

### Required Logging Components

1. **Sample Q&A Outputs** (CRITICAL)
   - Save generated Q&A samples during training at regular intervals
   - Must include: question type, market name, generated text, extracted answer, ground truth, correctness
   - Location: `/local/home/wangni/results/training_samples/samples_step_{N}.txt`
   - Frequency: Every 100 steps (or EVAL_EVERY)
   - Format: Human-readable text files for easy inspection

2. **Training Metrics** (CRITICAL)
   - Log to Wandb project: `opentslm-1101`
   - Required metrics:
     - `train/loss`: Per-step training loss
     - `train/lr`: Current learning rate
     - `train/epoch`: Current epoch
     - `val/accuracy`: Validation accuracy
     - `val/correct`: Number of correct predictions
     - `val/total`: Total validation samples evaluated

3. **Training Logs** (IMPORTANT)
   - Console output saved to file
   - Location: `/data/local/home/wangni/OpenTSLM/wandb/latest-run/files/output.log`
   - Must show: progress bars, epoch summaries, evaluation results

### Sample Output Format

**Critical**: Sample outputs must show the actual LLM generation, not just metrics.

```
================================================================================
Training Step 100 - Sample Outputs
================================================================================

Sample 1
--------------------------------------------------------------------------------
Type: past_trend
Market: Will Sean McDermott be the next coach fired?

Generated: decreasing

Extracted: decreasing
Ground Truth: decreasing
Correct: ✓

================================================================================
```

### Implementation Reference

See `train_model.py:91-165` for the `evaluate_model()` function that implements this logging.

**DO NOT skip sample logging** - it's essential for debugging and understanding model behavior.

## How to Run

### 0. Set Environment Variables (CRITICAL - Do This First!)

**ALWAYS run this before ANY Python script:**

```bash
export HF_HOME=/local/home/wangni/.cache/huggingface
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TRANSFORMERS_CACHE=/local/home/wangni/.cache/huggingface/transformers
```

### 1. Download Data

```bash
cd /local/home/wangni
export HF_HOME=/local/home/wangni/.cache/huggingface
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TRANSFORMERS_CACHE=/local/home/wangni/.cache/huggingface/transformers

conda run -n tslm python download_polymarket_enhanced.py > download_log.txt 2>&1
```

**Output**: `/local/home/wangni/polymarket_data/raw/market_price_data_enhanced.json`

### 2. Process Dataset (Automatic in Training)

The dataset is automatically processed during training:
- Class: `PolymarketTrendDataset` in `src/time_series_datasets/polymarket/PolymarketTrendDataset.py`
- Processed data: `/local/home/wangni/polymarket_data/processed/trend_dataset.json`

### 3. Test Untrained Encoder (Optional Baseline)

```bash
cd /data/local/home/wangni/OpenTSLM
export HF_HOME=/local/home/wangni/.cache/huggingface
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TRANSFORMERS_CACHE=/local/home/wangni/.cache/huggingface/transformers

CUDA_VISIBLE_DEVICES=0 conda run -n tslm python test_untrained_encoder.py \
  --num_samples 50 \
  --device cuda 2>&1 | tee /local/home/wangni/results/untrained_test.log
```

**Output**:
- Log: `/local/home/wangni/results/untrained_test.log`
- Samples: `/local/home/wangni/results/untrained_samples.txt`

### 4. Train Model (4 GPUs)

```bash
cd /data/local/home/wangni/OpenTSLM
export HF_HOME=/local/home/wangni/.cache/huggingface
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TRANSFORMERS_CACHE=/local/home/wangni/.cache/huggingface/transformers

bash run_training.sh
```

Or manually:

```bash
export HF_HOME=/local/home/wangni/.cache/huggingface
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TRANSFORMERS_CACHE=/local/home/wangni/.cache/huggingface/transformers

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master_addr=localhost \
  --master_port=29500 \
  train_model.py
```

**Outputs**:
- Training log: `/data/local/home/wangni/OpenTSLM/wandb/latest-run/files/output.log`
- Sample Q&As: `/local/home/wangni/results/training_samples/samples_step_{N}.txt` (CRITICAL)
- Checkpoints: `/local/home/wangni/results/polymarket_checkpoints/`
- Wandb: https://wandb.ai (project: `opentslm-1101`)

### 5. Monitor Training

**Check logs**:
```bash
tail -f /data/local/home/wangni/OpenTSLM/wandb/latest-run/files/output.log
```

**Check sample outputs**:
```bash
cat /local/home/wangni/results/training_samples/samples_step_100.txt
```

**Wandb dashboard**:
- Project: `opentslm-1101`
- Metrics: train/loss, val/accuracy, train/lr

### 6. Kill Training (If Needed)

```bash
pkill -f "train_model.py"
```

## File Locations

### Data
- Raw data: `/local/home/wangni/polymarket_data/raw/market_price_data_enhanced.json`
- Processed: `/local/home/wangni/polymarket_data/processed/trend_dataset.json`

### Code
- Dataset: `src/time_series_datasets/polymarket/PolymarketTrendDataset.py`
- Model: `src/model/llm/OpenTSLMSP.py`
- Training: `train_model.py`
- Testing: `test_untrained_encoder.py`

### Outputs
- Checkpoints: `/local/home/wangni/results/polymarket_checkpoints/`
- Training samples: `/local/home/wangni/results/training_samples/`
- Logs: `/data/local/home/wangni/OpenTSLM/wandb/`
- Results: `/local/home/wangni/results/`

### Documentation
- Dataset flow: `/local/home/wangni/results/DATASET_AND_EMBEDDING_FLOW.md`
- This guide: `/data/local/home/wangni/OpenTSLM/CLAUDE.md`

## Key Implementation Details

### Time Series Split Logic
```python
mid_point = len(time_series) // 2

if question_type == 'past_trend':
    ts_data = time_series[:mid_point]  # First half
    question = "What was the trend in the first period?"
    # Ground truth from first half slope
else:  # future_trend
    ts_data = time_series[:mid_point]  # Model sees ONLY first half!
    question = "What is the trend in the second period?"
    # Ground truth from SECOND half slope (model must predict)
```

### Answer Extraction (Fuzzy Parsing)
```python
def extract_answer(text: str) -> str:
    text_lower = text.lower()
    has_increasing = 'increasing' in text_lower or 'increase' in text_lower
    has_decreasing = 'decreasing' in text_lower or 'decrease' in text_lower

    if has_increasing and has_decreasing:
        # Return first occurrence
        inc_pos = text_lower.find('increas')
        dec_pos = text_lower.find('decreas')
        return 'increasing' if inc_pos < dec_pos else 'decreasing'
    elif has_increasing:
        return 'increasing'
    elif has_decreasing:
        return 'decreasing'
    else:
        return 'unknown'
```

### DDP Training Setup
- 4 GPUs (0,1,2,3)
- DistributedDataParallel wrapper
- DistributedSampler for training
- Gradient synchronization across GPUs
- Only rank 0 logs to wandb and saves checkpoints

## Environment

### ⚠️ CRITICAL: Always Use the `tslm` Conda Environment

**All Python scripts MUST be run in the `tslm` conda environment.**

```bash
# Activate environment (for interactive use)
conda activate tslm

# OR use conda run (for scripts)
conda run -n tslm python your_script.py
```

### Environment Details
- **Conda env**: `tslm` (REQUIRED)
- **Python**: 3.x
- **PyTorch**: With CUDA support
- **Transformers**: HuggingFace
- **Wandb**: For logging

### DO NOT:
- ❌ Run scripts in base environment
- ❌ Run scripts in other conda environments
- ❌ Use system Python

### Always use `conda run -n tslm` or activate `tslm` first!

## Common Issues

1. **Wrong conda environment** (CRITICAL)
   - ❌ **WRONG**: Running in base or other environment
   - ✅ **CORRECT**: Must use `tslm` environment
   - **Check**: `conda info --envs` and look for `*` next to `tslm`
   - **Solution**: Always use `conda activate tslm` or `conda run -n tslm`

2. **Files in wrong location** (CRITICAL)
   - ❌ **WRONG**: Files cached to `/home/wangni/.cache/`
   - ✅ **CORRECT**: All files must be in `/local/home/wangni/`
   - **Solution**: Always set `HF_HOME` and `TRANSFORMERS_CACHE` before running Python

3. **Model gated/not found**
   - Download and cache model to `/local/home/wangni/.cache/huggingface/`
   - Set `HF_TOKEN` environment variable if needed
   - Verify model is accessible before training

4. **Path confusion**
   - Data: `/local/home/wangni/` (symlink target)
   - Code: `/data/local/home/wangni/OpenTSLM/` (git repo)
   - Always use absolute paths in scripts

5. **GPU availability**
   - Training script uses 4 GPUs (0,1,2,3) by default
   - Adjust `CUDA_VISIBLE_DEVICES` if needed
   - Use `nvidia-smi` to check GPU usage

6. **Low accuracy early in training**
   - Normal - encoder is learning from scratch
   - Expect random outputs initially
   - Accuracy should improve after several epochs

7. **Missing sample outputs**
   - Verify `evaluate_model()` has `save_samples=True`
   - Check `/local/home/wangni/results/training_samples/` directory exists
   - Ensure evaluation runs every 100 steps

8. **Import errors or missing packages**
   - Make sure you're in the `tslm` environment (see #1)
   - All required packages should be installed in `tslm`
   - DO NOT install packages in other environments
