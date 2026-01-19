#!/bin/bash
# Launcher script for 4-GPU distributed training

set -e

echo "================================================================================"
echo "Starting OpenTSLM Training on 4 GPUs"
echo "================================================================================"

# CRITICAL: Set HuggingFace environment variables
export HF_HOME=/local/home/wangni/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:-YOUR_HF_TOKEN_HERE}"
export TRANSFORMERS_CACHE=/local/home/wangni/.cache/huggingface/transformers

# Set GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Number of GPUs
NGPUS=4

# Change to script directory
cd /data/local/home/wangni/OpenTSLM

echo "Configuration:"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"
echo "  Number of GPUs: $NGPUS"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo ""

# Run with torchrun (modern way) or fall back to python -m torch.distributed.launch
if command -v torchrun &> /dev/null; then
    echo "Using torchrun..."
    conda run -n tslm torchrun \
        --nproc_per_node=$NGPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_model.py
else
    echo "Using torch.distributed.launch..."
    conda run -n tslm python -m torch.distributed.launch \
        --nproc_per_node=$NGPUS \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        train_model.py
fi

echo ""
echo "================================================================================"
echo "Training Complete!"
echo "================================================================================"
