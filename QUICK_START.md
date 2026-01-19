# Quick Start: Enhanced Logging System

## Run Training with Enhanced Logging

```bash
cd /data/local/home/wangni/OpenTSLM
conda activate tslm
export CUDA_VISIBLE_DEVICES=0

# Run training
python train_polymarket_enhanced.py
```

## View Training Results

### Option 1: Interactive Viewer
```bash
# View latest session
python visualize_logs.py --latest

# Then use menu:
# 1 - Training summary (losses, ASCII chart)
# 2 - Evaluation summary (accuracy by question type)
# 3 - Browse all samples (with prompts & answers)
# 6 - Export markdown report
```

### Option 2: Direct File Access
```bash
# Find your session
ls /local/home/wangni/results/polymarket_enhanced_logs/

# View files
cd /local/home/wangni/results/polymarket_enhanced_logs/<SESSION_ID>

# Training logs (per step)
cat training_log.jsonl | jq .

# Evaluation logs (per sample)
cat evaluation_log.jsonl | jq .

# Final metrics
cat metrics.json | jq .

# Configuration
cat config.json | jq .
```

### Option 3: Export Report
```bash
# Generate markdown report
python visualize_logs.py --latest --export

# View report
cat /local/home/wangni/results/polymarket_enhanced_logs/<SESSION_ID>/report.md
```

## Key Features

### 1. More Training Data (3x)
The enhanced script uses `PolymarketQADatasetExpanded` which generates:
- 3 time windows per market (full, early, mid)
- 9 question types vs 5 in basic loader
- 48 samples from 2 markets vs 16 samples

### 2. Structured JSON Logs
All logs are in JSON format for easy parsing:
```python
import json

# Read training logs
with open('training_log.jsonl') as f:
    for line in f:
        log = json.loads(line)
        print(f"Epoch {log['epoch']}, Loss: {log['loss']}")

# Read evaluation logs
with open('evaluation_log.jsonl') as f:
    for line in f:
        sample = json.loads(line)
        print(f"Question: {sample['post_prompt']}")
        print(f"Answer: {sample['generated_answer']}")
        print(f"Correct: {sample['is_correct']}")
```

### 3. Detailed Prompt Tracking
Each evaluation log includes:
- Full pre-prompt text
- Time series descriptions
- Number of patches (embedding tokens)
- Post-prompt/question
- Expected answer
- Generated answer
- Correctness

## Customization

### Change Number of Epochs
Edit `train_polymarket_enhanced.py`:
```python
NUM_EPOCHS = 10  # Change from 5 to 10
```

### Change Model
Edit `train_polymarket_enhanced.py`:
```python
model = OpenTSLMSP(llm_id="gpt2", device=DEVICE)
# Change to any HuggingFace model
```

### Use More Real Polymarket Data
Edit `/local/home/wangni/download_polymarket_simple.py`:
```python
for i, market in enumerate(markets[:100]):  # Increase limit
    if len(successful) >= 50:  # Increase successful count
```

Then run:
```bash
python /local/home/wangni/download_polymarket_simple.py
```

## Files

### Created Files
- `train_polymarket_enhanced.py` - Enhanced training script
- `visualize_logs.py` - Interactive log viewer
- `ENHANCED_LOGGING_SUMMARY.md` - Detailed documentation
- `QUICK_START.md` - This file

### Log Files (per session)
- `config.json` - Training configuration
- `metrics.json` - Final metrics
- `training_log.jsonl` - Training step logs
- `evaluation_log.jsonl` - Evaluation sample logs
- `report.md` - Auto-generated report (if exported)

## Example Session

```bash
# 1. Run training
$ python train_polymarket_enhanced.py

====================================================================================================
                        ENHANCED POLYMARKET TRAINING WITH STRUCTURED LOGGING
====================================================================================================

📁 Session ID: 20251028_094844
📁 Logs directory: /local/home/wangni/results/polymarket_enhanced_logs/20251028_094844

1️⃣  Initializing OpenTSLMSP model on cuda...
✅ Model initialized

2️⃣  Loading Polymarket datasets (EXPANDED version)...
✅ Train: 38 samples
✅ Val: 10 samples
✅ Test: 0 samples
📈 Total samples: 48
   (3x more than basic loader due to multiple time windows)

[Training runs for 5 epochs...]

====================================================================================================
                                           FINAL RESULTS
====================================================================================================

✅ Accuracy: 0.00% (0/0 correct)
📈 Final Train Loss: 8.6651

📊 All logs saved to: /local/home/wangni/results/polymarket_enhanced_logs/20251028_094844

💡 Use visualize_logs.py to view results interactively!

# 2. View results
$ python visualize_logs.py --latest

[Interactive menu appears...]

# 3. Export report
$ python visualize_logs.py --latest --export
✅ Report exported to: .../report.md
```

## Troubleshooting

### Issue: Import Error
```bash
# Make sure you're in the OpenTSLM directory
cd /data/local/home/wangni/OpenTSLM

# Make sure conda environment is activated
conda activate tslm
```

### Issue: CUDA Out of Memory
```bash
# Use a specific GPU
export CUDA_VISIBLE_DEVICES=0

# Or reduce batch size in train_polymarket_enhanced.py
BATCH_SIZE = 1  # Change from 2
```

### Issue: Model Access Error
The script uses GPT-2 which is fully open. If you want to use other models:
```python
# For Llama models, you need HuggingFace authentication
huggingface-cli login

# Or use another open model
model = OpenTSLMSP(llm_id="EleutherAI/gpt-neo-125m", device=DEVICE)
```

## Next Steps

1. Run the enhanced training script
2. Explore logs with the visualization tool
3. Experiment with different models
4. Download more polymarket data
5. Customize logging for your specific needs

For detailed documentation, see `ENHANCED_LOGGING_SUMMARY.md`
