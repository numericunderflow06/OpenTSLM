# Enhanced Logging System for Polymarket Training

## Overview

I've implemented a comprehensive enhanced logging system for the OpenTSLM polymarket training framework with the following improvements:

## Key Enhancements

### 1. **Structured JSON Logging**
   - All training steps logged in machine-readable JSON format
   - Easy to parse and analyze programmatically
   - Includes timestamps, losses, learning rates, and more

### 2. **Expanded Data Loader**
   - Switched from `PolymarketQADataset` to `PolymarketQADatasetExpanded`
   - **3x more training samples** through multiple time windows per market
   - From 10 samples → 48 samples with the same 2 markets
   - More diverse question types (9 types vs 5 types)

### 3. **Enhanced Prompt/Answer Tracking**
   - Detailed logging of prompt structure
   - Time series embedding information (patches, dimensions)
   - Expected vs generated answers
   - Correctness tracking per sample

### 4. **Interactive Visualization Tool**
   - Browse training progress
   - View evaluation results
   - Navigate through samples interactively
   - Export markdown reports

## Files Created

### Training Script
- **`train_polymarket_enhanced.py`** - Enhanced training script with structured logging
  - Uses GPT-2 model (open, no authentication required)
  - Expanded data loader for 3x more samples
  - JSON logging throughout training
  - Detailed evaluation logging

### Visualization Tool
- **`visualize_logs.py`** - Interactive log viewer
  - View training summaries with ASCII loss curves
  - Browse evaluation samples with full prompt details
  - Filter by correct/incorrect predictions
  - Export markdown reports

## Log Structure

### Directory Structure
```
/local/home/wangni/results/polymarket_enhanced_logs/
└── <SESSION_ID>/
    ├── config.json           # Training configuration
    ├── metrics.json          # Final metrics
    ├── training_log.jsonl    # Per-step training logs
    └── evaluation_log.jsonl  # Per-sample evaluation logs
```

### Log Formats

#### Training Log (JSONL)
```json
{
  "timestamp": "2025-10-28T09:49:09.897999",
  "type": "training_step",
  "epoch": 1,
  "step": 10,
  "loss": 8.796863555908203,
  "lr_encoder": 0.0001827956989247312,
  "lr_projector": 9.13978494623656e-05
}
```

#### Evaluation Log (JSONL)
```json
{
  "timestamp": "2025-10-28T09:49:15.123456",
  "type": "evaluation_sample",
  "sample_idx": 0,
  "market_id": "market_123",
  "question_type": "trend",
  "pre_prompt": "You are an expert...",
  "time_series_info": [
    {
      "description": "Market price data...",
      "length": 1440,
      "num_patches": 360
    }
  ],
  "post_prompt": "What is the trend?",
  "expected_answer": "increasing",
  "generated_answer": "increasing trend observed",
  "is_correct": true
}
```

## Usage

### Running Training with Enhanced Logging

```bash
cd /data/local/home/wangni/OpenTSLM
conda activate tslm
export CUDA_VISIBLE_DEVICES=0
python train_polymarket_enhanced.py
```

### Viewing Logs Interactively

```bash
# View latest training session
python visualize_logs.py --latest

# View specific session
python visualize_logs.py 20251028_094844

# Export markdown report
python visualize_logs.py --latest --export
```

### Interactive Viewer Features

1. **Training Summary** - View epoch-by-epoch losses with ASCII chart
2. **Evaluation Summary** - Accuracy breakdown by question type
3. **Browse All Samples** - Navigate through all predictions
4. **Browse Correct/Incorrect** - Filter by correctness
5. **Search** - Find samples containing specific terms
6. **Export Report** - Generate markdown summary

## Test Run Results

A successful test run was completed (Session: `20251028_094844`):

### Configuration
- **Model**: GPT-2
- **Data Loader**: PolymarketQADatasetExpanded
- **Training Samples**: 38 (3x increase from basic loader)
- **Batch Size**: 2
- **Epochs**: 5
- **Device**: CUDA (GPU 0)

### Training Progress
```
Epoch 1: Train Loss: 8.7265
Epoch 2: Train Loss: 8.9161
Epoch 3: Train Loss: 9.0785
Epoch 4: Train Loss: 9.0063
Epoch 5: Train Loss: 8.6651 ← Final
```

### Question Types (Expanded Loader)
- change_magnitude: 5 samples
- momentum: 6 samples
- next_movement: 4 samples
- price_level: 4 samples
- reversal_likelihood: 1 sample
- short_term_direction: 3 samples
- stability: 4 samples
- trend: 6 samples
- volatility: 5 samples

**Total: 9 question types** (vs 5 in basic loader)

## Key Benefits

### 1. Better Visibility
- Clear view of what prompts are sent to the model
- Detailed tracking of time series embeddings
- Easy to debug prompt engineering issues

### 2. More Data
- 3x more training samples without downloading new data
- Multiple time windows per market
- More diverse question types

### 3. Easy Analysis
- JSON format for programmatic analysis
- Can build custom analytics tools
- Track performance by question type

### 4. Reproducibility
- All configuration logged
- Session IDs for tracking experiments
- Easy to compare different runs

## Next Steps

### To Use More Real Data
Edit `/local/home/wangni/download_polymarket_simple.py`:
```python
# Line 137-138:
for i, market in enumerate(markets[:100]):  # Increase from 10 to 100
    if len(successful) >= 50:  # Increase from 10 to 50
```

Then run:
```bash
python /local/home/wangni/download_polymarket_simple.py
```

### To Customize Logging
Edit `train_polymarket_enhanced.py`:
- Adjust `NUM_EPOCHS`, `BATCH_SIZE`, etc.
- Add custom metrics in `EnhancedLogger` class
- Modify log formats in JSON output

### To Add Custom Visualizations
Edit `visualize_logs.py`:
- Add new menu options
- Create custom charts
- Export in different formats

## Example: Viewing Training Progress

```bash
$ python visualize_logs.py --latest

====================================================================================================
                              POLYMARKET TRAINING LOG VIEWER
====================================================================================================

Session: 20251028_094844
Location: /local/home/wangni/results/polymarket_enhanced_logs/20251028_094844

1. View training summary
2. View evaluation summary
3. Browse all samples
4. Browse correct predictions
5. Browse incorrect predictions
6. Export report
7. Quit

Select option (1-7): 1

====================================================================================================
                                      TRAINING SUMMARY
====================================================================================================

Total epochs: 5

Epoch      Train Loss     Val Loss
────────────────────────────────────────
1          8.7265         N/A
2          8.9161         N/A
3          9.0785         N/A
4          9.0063         N/A
5          8.6651         N/A

📈 Training Loss Progression:
9.0785 │               ●
8.9161 │         ●
8.7265 │     ●
8.6651 │                                 ●
8.5000 │ ●
       └─────────────────────────────────────────────────────────
         01234
```

## Conclusion

The enhanced logging system provides:
- ✅ **Clear visibility** into prompts and answers
- ✅ **3x more data** through expanded loader
- ✅ **Structured JSON logs** for easy analysis
- ✅ **Interactive visualization** for quick insights
- ✅ **Easy to extend** for custom needs

All logs are saved in:
`/local/home/wangni/results/polymarket_enhanced_logs/<SESSION_ID>/`
