# OpenTSLM Polymarket Training Pipeline

## Overview

This pipeline trains an OpenTSLM encoder on Polymarket trend prediction with the following specifications:

- **Dataset**: 1000 Polymarket markets, 12 months of data each
- **Task**: Binary trend classification (increasing/decreasing)
- **Questions**: 2 per market (past trend + future forecast)
- **Training**: 4-GPU DDP with encoder + projector training only (LLM frozen)
- **Logging**: Wandb project `opentslm-1101`

## What Was Created

### 1. Data Download & Processing

- **`/local/home/wangni/download_polymarket_enhanced.py`**
  - Downloads 12 months of data for up to 1000 markets
  - Saves to `/local/home/wangni/polymarket_data/raw/`

- **`PolymarketTrendDataset.py`**
  - Splits time series in half
  - Computes linear regression slopes
  - Creates past/future trend questions
  - Excludes markets with slope=0

### 2. Dataset Format

Each sample contains:
- **Time series**: Split in half
- **Past trend question**: "What was the trend in the first period?"
- **Future forecast**: "What is the trend in the second period?"
- **Prompt format**: "Answer with ONLY one word: increasing or decreasing"
- **Ground truth**: Computed via linear regression slope

### 3. Testing & Training Scripts

- **`test_untrained_encoder.py`**
  - Tests untrained encoder on 50 samples
  - Logs questions/answers to files
  - Uses fuzzy parsing to extract answers
  - Saves results to `/local/home/wangni/results/untrained_test/`

- **`train_model.py`**
  - 4-GPU DDP training
  - Trains encoder + projector only (LLM frozen)
  - Wandb logging (project: opentslm-1101)
  - Evaluates every 100 steps
  - Saves checkpoints to `/local/home/wangni/results/polymarket_checkpoints/`

### 4. Launcher Scripts

- **`run_training.sh`**
  - Launches 4-GPU distributed training
  - Uses CUDA_VISIBLE_DEVICES=0,1,2,3
  - Runs with torchrun or torch.distributed.launch

- **`run_pipeline.sh`**
  - Master orchestration script
  - Waits for download
  - Processes data
  - Runs untrained test
  - Launches training in background

## How to Use

### Quick Start

```bash
cd /data/local/home/wangni/OpenTSLM
bash run_pipeline.sh
```

This will:
1. Wait for download to complete (already running)
2. Process the raw data
3. Test untrained encoder
4. Launch 4-GPU training

### Manual Steps

If you want to run steps individually:

```bash
# 1. Check download status
ls -lh /local/home/wangni/polymarket_data/raw/market_price_data_enhanced.json

# 2. Process data
cd /data/local/home/wangni/OpenTSLM
conda run -n tslm python -m src.time_series_datasets.polymarket.PolymarketTrendDataset

# 3. Test untrained encoder
conda run -n tslm python test_untrained_encoder.py --num_samples 50

# 4. Launch training
bash run_training.sh
```

### Monitor Training

```bash
# View training log
tail -f /local/home/wangni/results/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# View wandb
# Go to https://wandb.ai and look for project "opentslm-1101"
```

## Output Locations

- **Raw data**: `/local/home/wangni/polymarket_data/raw/`
- **Processed data**: `/local/home/wangni/polymarket_data/processed/`
- **Untrained test results**: `/local/home/wangni/results/untrained_test/`
- **Training checkpoints**: `/local/home/wangni/results/polymarket_checkpoints/`
- **Training logs**: `/local/home/wangni/results/training.log`

## Configuration

### Training Hyperparameters

In `train_model.py`:
- Batch size: 4 per GPU
- Epochs: 20
- Encoder LR: 2e-4
- Projector LR: 1e-4
- Weight decay: 1e-2
- Warmup: 3% of total steps
- Gradient clipping: 1.0

### Model

- **LLM**: google/gemma-2-2b (frozen)
- **Encoder**: TransformerCNN (trainable)
- **Projector**: MLP (trainable)
- **Patch size**: 4

## Expected Results

### Untrained Encoder
- Baseline accuracy: ~50% (random guessing)
- Format following: Variable (model may not follow output format)

### Trained Encoder
- Target accuracy: >60-70% (depending on data quality)
- Better format following expected
- Improved trend recognition

## Troubleshooting

### Download Issues
```bash
# Check download progress
ps aux | grep download_polymarket_enhanced
tail -f /local/home/wangni/download_log.txt
```

### Training Issues
```bash
# Check GPU availability
nvidia-smi

# View training logs
tail -f /local/home/wangni/results/training.log

# Kill training if needed
pkill -f train_model.py
```

### Data Processing Issues
```bash
# Manually process data
cd /data/local/home/wangni/OpenTSLM
conda run -n tslm python -m src.time_series_datasets.polymarket.PolymarketTrendDataset
```

## Next Steps After Training

1. **Evaluate trained model**:
   ```bash
   conda run -n tslm python test_untrained_encoder.py \
       --num_samples 200 \
       --output_dir /local/home/wangni/results/trained_test
   ```
   (After loading the trained checkpoint in the script)

2. **Compare results**:
   - Untrained: `/local/home/wangni/results/untrained_test/`
   - Trained: `/local/home/wangni/results/trained_test/`

3. **Analyze wandb logs**:
   - Loss curves
   - Validation accuracy over time
   - Learning rate schedule

## File Structure

```
/data/local/home/wangni/OpenTSLM/
├── train_model.py                    # Main training script
├── test_untrained_encoder.py         # Testing script
├── run_training.sh                   # 4-GPU launcher
├── run_pipeline.sh                   # Master orchestrator
└── src/time_series_datasets/polymarket/
    └── PolymarketTrendDataset.py    # Dataset class

/local/home/wangni/
├── polymarket_data/
│   ├── raw/                          # Downloaded data
│   └── processed/                    # Processed dataset
├── results/
│   ├── untrained_test/              # Untrained encoder results
│   ├── polymarket_checkpoints/      # Training checkpoints
│   └── training.log                 # Training log
└── download_polymarket_enhanced.py   # Download script
```

## Questions?

The pipeline is designed to be fully automated. Just run `bash run_pipeline.sh` and it will handle everything.

For monitoring:
- Training progress: `tail -f /local/home/wangni/results/training.log`
- Wandb metrics: https://wandb.ai (project: opentslm-1101)
- GPU usage: `watch -n 1 nvidia-smi`
