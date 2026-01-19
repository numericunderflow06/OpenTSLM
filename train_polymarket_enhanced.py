#!/usr/bin/env python3
"""
Enhanced Polymarket Training with Structured Logging

Features:
- JSON-formatted logs for easy analysis
- Detailed prompt/answer tracking
- Real-time progress monitoring
- Uses expanded data loader for more samples
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
import json
from datetime import datetime
from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from time_series_datasets.polymarket.PolymarketQADatasetExpanded import PolymarketQADatasetExpanded
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from model.llm.OpenTSLMSP import OpenTSLMSP
from model_config import PATCH_SIZE

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 5
LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_FRAC = 0.03

# Create logs directory
LOG_DIR = Path("/local/home/wangni/results/polymarket_enhanced_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Session ID for this run
SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_LOG_DIR = LOG_DIR / SESSION_ID
SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log files
TRAINING_LOG = SESSION_LOG_DIR / "training_log.jsonl"
EVALUATION_LOG = SESSION_LOG_DIR / "evaluation_log.jsonl"
METRICS_LOG = SESSION_LOG_DIR / "metrics.json"
CONFIG_LOG = SESSION_LOG_DIR / "config.json"


class EnhancedLogger:
    """Enhanced logger with structured JSON output."""

    def __init__(self, session_dir):
        self.session_dir = session_dir
        self.training_log = session_dir / "training_log.jsonl"
        self.evaluation_log = session_dir / "evaluation_log.jsonl"

    def log_training_step(self, epoch, step, loss, lr_encoder, lr_projector):
        """Log a training step."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "training_step",
            "epoch": epoch,
            "step": step,
            "loss": float(loss),
            "lr_encoder": float(lr_encoder),
            "lr_projector": float(lr_projector)
        }
        with open(self.training_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_epoch_summary(self, epoch, train_loss, val_loss=None):
        """Log epoch summary."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "epoch_summary",
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss) if val_loss else None
        }
        with open(self.training_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def log_evaluation_sample(self, sample_idx, sample_data, prediction, expected, is_correct):
        """Log detailed evaluation sample."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "evaluation_sample",
            "sample_idx": sample_idx,
            "market_id": sample_data.get('market_id', 'N/A'),
            "question_type": sample_data.get('question_type', 'N/A'),
            "pre_prompt": sample_data['pre_prompt'],
            "time_series_info": [
                {
                    "description": ts_text,
                    "length": len(ts_data),
                    "num_patches": len(ts_data) // PATCH_SIZE
                }
                for ts_text, ts_data in zip(sample_data['time_series_text'], sample_data['time_series'])
            ],
            "post_prompt": sample_data['post_prompt'],
            "expected_answer": expected,
            "generated_answer": prediction,
            "is_correct": is_correct
        }
        with open(self.evaluation_log, 'a') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


def print_header(text, char="="):
    """Print a formatted header."""
    print(f"\n{char*100}")
    print(f"{text.center(100)}")
    print(f"{char*100}\n")


def main():
    print_header("ENHANCED POLYMARKET TRAINING WITH STRUCTURED LOGGING")

    # Save configuration
    config = {
        "session_id": SESSION_ID,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "lr_encoder": LR_ENCODER,
        "lr_projector": LR_PROJECTOR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_fraction": WARMUP_FRAC,
        "patch_size": PATCH_SIZE,
        "model": "gpt2",
        "data_loader": "PolymarketQADatasetExpanded",
        "log_dir": str(SESSION_LOG_DIR)
    }

    with open(CONFIG_LOG, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"📁 Session ID: {SESSION_ID}")
    print(f"📁 Logs directory: {SESSION_LOG_DIR}")
    print(f"📝 Config saved to: {CONFIG_LOG}")

    # Initialize logger
    logger = EnhancedLogger(SESSION_LOG_DIR)

    # Initialize model
    print(f"\n1️⃣  Initializing OpenTSLMSP model on {DEVICE}...")
    model = OpenTSLMSP(llm_id="gpt2", device=DEVICE).to(DEVICE)
    print(f"✅ Model initialized")

    # Load datasets (using EXPANDED loader for more data)
    print(f"\n2️⃣  Loading Polymarket datasets (EXPANDED version)...")
    train_dataset = PolymarketQADatasetExpanded("train", model.get_eos_token())
    val_dataset = PolymarketQADatasetExpanded("validation", model.get_eos_token())
    test_dataset = PolymarketQADatasetExpanded("test", model.get_eos_token())

    print(f"✅ Train: {len(train_dataset)} samples")
    print(f"✅ Val: {len(val_dataset)} samples")
    print(f"✅ Test: {len(test_dataset)} samples")
    print(f"📈 Total samples: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    print(f"   (3x more than basic loader due to multiple time windows)")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Batch size 1 for detailed logging
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    # Setup optimizer
    enc_params = list(model.encoder.parameters())
    proj_params = list(model.projector.projector.parameters())

    optimizer = AdamW([
        {"params": enc_params, "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY},
        {"params": proj_params, "lr": LR_PROJECTOR, "weight_decay": WEIGHT_DECAY},
    ])

    # Setup scheduler
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\n3️⃣  Training configuration:")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Training batches per epoch: {len(train_loader)}")

    # Training loop
    print_header("STARTING TRAINING", "=")

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'─'*100}")
        print(f"EPOCH {epoch}/{NUM_EPOCHS}")
        print(f"{'─'*100}")

        # Training
        model.train()
        running_loss = 0.0

        prog = tqdm(train_loader, desc=f"Training")
        for batch_idx, batch in enumerate(prog):
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            # Log training step
            if global_step % 10 == 0:
                logger.log_training_step(
                    epoch=epoch,
                    step=global_step,
                    loss=loss.item(),
                    lr_encoder=optimizer.param_groups[0]['lr'],
                    lr_projector=optimizer.param_groups[1]['lr']
                )

            prog.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        logger.log_epoch_summary(epoch, avg_train_loss)

        print(f"\n✅ Epoch {epoch} completed - Train Loss: {avg_train_loss:.4f}")

    # Final evaluation with detailed logging
    print_header("FINAL EVALUATION - DETAILED LOGGING", "=")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Generate prediction
            predictions = model.generate(batch, max_new_tokens=50)

            # Log details for each sample
            for sample, pred in zip(batch, predictions):
                expected = sample['answer'].replace(model.get_eos_token(), '').strip()
                generated = pred.replace(model.get_eos_token(), '').strip()

                # Check correctness (fuzzy matching)
                is_correct = expected.lower() in generated.lower()

                # Log to JSON
                logger.log_evaluation_sample(
                    sample_idx=idx,
                    sample_data=sample,
                    prediction=generated,
                    expected=expected,
                    is_correct=is_correct
                )

                if is_correct:
                    correct += 1
                total += 1

                # Print first 3 samples in detail
                if idx < 3:
                    print(f"\n{'─'*80}")
                    print(f"Sample {idx + 1}")
                    print(f"{'─'*80}")
                    print(f"Market: {sample.get('market_id', 'N/A')}")
                    print(f"Question Type: {sample.get('question_type', 'N/A')}")
                    print(f"\n🎯 Expected: {expected}")
                    print(f"🤖 Generated: {generated}")
                    print(f"{'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

    accuracy = correct / total if total > 0 else 0

    # Save final metrics
    final_metrics = {
        "session_id": SESSION_ID,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "final_train_loss": avg_train_loss,
        "num_epochs": NUM_EPOCHS,
        "timestamp": datetime.now().isoformat()
    }

    with open(METRICS_LOG, 'w') as f:
        json.dump(final_metrics, f, indent=2)

    print_header("FINAL RESULTS", "=")
    print(f"✅ Accuracy: {accuracy:.2%} ({correct}/{total} correct)")
    print(f"📈 Final Train Loss: {avg_train_loss:.4f}")
    print(f"\n📊 All logs saved to: {SESSION_LOG_DIR}")
    print(f"   • Training log: {TRAINING_LOG}")
    print(f"   • Evaluation log: {EVALUATION_LOG}")
    print(f"   • Metrics: {METRICS_LOG}")
    print(f"   • Config: {CONFIG_LOG}")
    print(f"\n💡 Use visualize_logs.py to view results interactively!")
    print_header("TRAINING COMPLETED", "=")


if __name__ == "__main__":
    main()
