#!/bin/bash
# Master pipeline script to orchestrate the complete workflow

set -e

echo "================================================================================"
echo "OpenTSLM Polymarket Training Pipeline"
echo "================================================================================"

PROJECT_DIR="/data/local/home/wangni/OpenTSLM"
RESULTS_DIR="/local/home/wangni/results"

cd $PROJECT_DIR

# Step 1: Check if download is complete
echo ""
echo "[1/5] Checking download status..."
echo "--------------------------------------------------------------------------------"

DOWNLOAD_FILE="/local/home/wangni/polymarket_data/raw/market_price_data_enhanced.json"

if [ ! -f "$DOWNLOAD_FILE" ]; then
    echo "⏳ Waiting for download to complete..."
    echo "   Target file: $DOWNLOAD_FILE"

    # Wait for download to complete (check every 30 seconds)
    while [ ! -f "$DOWNLOAD_FILE" ]; do
        sleep 30
        echo "   Still waiting..."
    done

    echo "✓ Download complete!"
else
    echo "✓ Download already complete!"
fi

# Check file size
FILE_SIZE=$(stat -f%z "$DOWNLOAD_FILE" 2>/dev/null || stat -c%s "$DOWNLOAD_FILE" 2>/dev/null)
echo "   File size: $(numfmt --to=iec-i --suffix=B $FILE_SIZE 2>/dev/null || echo $FILE_SIZE bytes)"

# Step 2: Process the data
echo ""
echo "[2/5] Processing raw data..."
echo "--------------------------------------------------------------------------------"

PROCESSED_FILE="/local/home/wangni/polymarket_data/processed/trend_dataset.json"

if [ ! -f "$PROCESSED_FILE" ]; then
    echo "Running data processing..."
    conda run -n tslm python -m src.time_series_datasets.polymarket.PolymarketTrendDataset
    echo "✓ Data processing complete!"
else
    echo "✓ Processed data already exists!"
fi

# Step 3: Test untrained encoder
echo ""
echo "[3/5] Testing untrained encoder..."
echo "--------------------------------------------------------------------------------"

UNTRAINED_RESULTS="$RESULTS_DIR/untrained_test/untrained_encoder_results.json"

if [ ! -f "$UNTRAINED_RESULTS" ]; then
    echo "Running untrained encoder test..."
    conda run -n tslm python test_untrained_encoder.py --num_samples 50
    echo "✓ Untrained encoder test complete!"
    echo "   Results saved to: $RESULTS_DIR/untrained_test/"
else
    echo "✓ Untrained test already complete!"
    echo "   Results: $UNTRAINED_RESULTS"
fi

# Step 4: Make scripts executable
echo ""
echo "[4/5] Preparing training scripts..."
echo "--------------------------------------------------------------------------------"

chmod +x run_training.sh
echo "✓ Training script ready"

# Step 5: Launch training
echo ""
echo "[5/5] Launching 4-GPU training..."
echo "--------------------------------------------------------------------------------"

echo "Training will run in the background with nohup"
echo "Log file: $RESULTS_DIR/training.log"
echo "Wandb project: opentslm-1101"
echo ""

# Create results directory
mkdir -p $RESULTS_DIR

# Launch training in background
nohup bash run_training.sh > $RESULTS_DIR/training.log 2>&1 &
TRAIN_PID=$!

echo "✓ Training launched!"
echo "   PID: $TRAIN_PID"
echo "   Log: tail -f $RESULTS_DIR/training.log"
echo "   Wandb: https://wandb.ai"
echo ""

echo "================================================================================"
echo "Pipeline Started Successfully!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Monitor training: tail -f $RESULTS_DIR/training.log"
echo "  2. View metrics: https://wandb.ai (project: opentslm-1101)"
echo "  3. Check checkpoints: /local/home/wangni/results/polymarket_checkpoints/"
echo ""
echo "Training process ID: $TRAIN_PID"
echo "To stop training: kill $TRAIN_PID"
echo ""
