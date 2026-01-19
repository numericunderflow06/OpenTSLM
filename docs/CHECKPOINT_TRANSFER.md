# Checkpoint Transfer Guide for Merged Training

This document explains how to transfer training checkpoints between clusters to resume training.

## Checkpoint Location

The merged training saves checkpoints to:

```
results_merged/{LLM_ID_SAFE}/{MODEL_TYPE}/checkpoints/
```

For the current training run (Llama-3.2-1B with OpenTSLMFlamingo):

```
results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints/
├── best_model.pt      # Main checkpoint file (~13GB)
└── loss_history.txt   # Training loss history
```

### Checkpoint Contents

The `best_model.pt` file contains:
- `encoder_state`: TransformerCNNEncoder weights
- `projector_state`: MLPProjector weights
- `base_model_state`: Flamingo/LLM weights (or LoRA adapter states)
- `optimizer_state`: AdamW optimizer state
- `scheduler_state`: Learning rate scheduler state
- `epoch`: Last completed epoch number
- `val_loss`: Best validation loss achieved

## Resuming Training on Another Cluster

### 1. Directory Structure Required

On the target cluster, create the same directory structure:

```bash
mkdir -p results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints
```

### 2. Place Checkpoint Files

Copy these files to the target cluster:

```
results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints/
├── best_model.pt
└── loss_history.txt  # Optional, for logging continuity
```

### 3. Resume Training Command

Run the training script - it will automatically detect and load the checkpoint:

```bash
# Single GPU
python merged_training.py --model OpenTSLMFlamingo --num_epochs 30

# Multi-GPU with torchrun
torchrun --nproc_per_node=8 merged_training.py --model OpenTSLMFlamingo --num_epochs 30
```

The script will print:
```
Resuming from epoch 9 (val_loss: 0.4685)
```

### 4. Evaluation Only Mode

To run evaluation without training:

```bash
python merged_training.py --model OpenTSLMFlamingo --eval_only
```

## Transfer Instructions

### From Current Cluster to Your Computer

```bash
# On your local computer, run:
scp -r <username>@<current-cluster>:/local/home/wangni/OpenTSLM/results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints ./opentslm_checkpoint/

# Or use rsync for better resume capability (recommended for large files):
rsync -avP <username>@<current-cluster>:/local/home/wangni/OpenTSLM/results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints/ ./opentslm_checkpoint/
```

### From Your Computer to Target Cluster

```bash
# 1. Create directory on target cluster
ssh <username>@<target-cluster> "mkdir -p /path/to/OpenTSLM/results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints"

# 2. Transfer checkpoint
rsync -avP ./opentslm_checkpoint/ <username>@<target-cluster>:/path/to/OpenTSLM/results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints/

# Or with scp:
scp -r ./opentslm_checkpoint/* <username>@<target-cluster>:/path/to/OpenTSLM/results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints/
```

## Verification

After transfer, verify the checkpoint on the target cluster:

```bash
# Check file size (should be ~13GB)
ls -lh results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints/best_model.pt

# Verify checkpoint contents
python -c "
import torch
ckpt = torch.load('results_merged/Llama_3_2_1B/OpenTSLMFlamingo/checkpoints/best_model.pt', map_location='cpu', weights_only=False)
print('Checkpoint keys:', list(ckpt.keys()))
print('Epoch:', ckpt.get('epoch'))
print('Val loss:', ckpt.get('val_loss'))
"
```

Expected output:
```
Checkpoint keys: ['encoder_state', 'projector_state', 'base_model_state', 'optimizer_state', 'scheduler_state', 'epoch', 'val_loss']
Epoch: 9
Val loss: 0.4684699773788452
```

## Training Status Summary

| Metric | Value |
|--------|-------|
| Epochs completed | 9 |
| Best validation loss | 0.4685 |
| Convergence status | **Not converged** (min 10 epochs required) |
| Checkpoint size | ~13GB |

### Last Training Loss History

| Epoch | Train Loss | Val Loss |
|-------|------------|----------|
| 7 | 0.516 | 0.489 |
| 8 | 0.502 | 0.475 |
| 9 | 0.491 | 0.468 |

Training was still improving when stopped. Continuing for more epochs is recommended.

## Notes

- The checkpoint automatically resumes optimizer and scheduler states, so learning rate scheduling continues correctly
- If you want to start fresh (ignore checkpoint), rename or delete the `checkpoints` directory
- The `loss_history.txt` is for human reference only; it's not required for resuming
- Make sure the target cluster has the same PyTorch version to avoid compatibility issues
