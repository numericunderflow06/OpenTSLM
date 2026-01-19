#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

"""
Train Polymarket dataset with detailed logging to show:
1. Full question structure (with time series embedding markers)
2. Model's generated answer
3. Expected answer
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm

from time_series_datasets.polymarket.PolymarketQADataset import PolymarketQADataset
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from model.llm.OpenTSLMSP import OpenTSLMSP
from model_config import PATCH_SIZE

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 3
LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_FRAC = 0.03

print("="*100)
print("POLYMARKET TRAINING WITH DETAILED LOGGING")
print("="*100)

# Initialize model
print(f"\n1. Initializing OpenTSLMSP model on {DEVICE}...")
model = OpenTSLMSP(llm_id="meta-llama/Llama-3.2-1B", device=DEVICE).to(DEVICE)
print(f"✅ Model initialized")

# Load datasets
print(f"\n2. Loading Polymarket datasets...")
train_dataset = PolymarketQADataset("train", model.get_eos_token())
val_dataset = PolymarketQADataset("validation", model.get_eos_token())
test_dataset = PolymarketQADataset("test", model.get_eos_token())

print(f"✅ Train: {len(train_dataset)} samples")
print(f"✅ Val: {len(val_dataset)} samples")
print(f"✅ Test: {len(test_dataset)} samples")

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

print(f"\n3. Training configuration:")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Total steps: {total_steps}")
print(f"   Warmup steps: {warmup_steps}")


def log_sample_with_details(sample, prediction, idx):
    """Log detailed information about a sample."""
    print(f"\n{'='*100}")
    print(f"SAMPLE {idx + 1} - Detailed Breakdown")
    print(f"{'='*100}")

    # Market info
    print(f"\n📊 Market Information:")
    print(f"   Market ID: {sample.get('market_id', 'N/A')}")
    print(f"   Question Type: {sample.get('question_type', 'N/A')}")

    # Full prompt structure
    print(f"\n📝 FULL PROMPT SENT TO MODEL:")
    print(f"{'─'*100}")

    # Pre-prompt
    print(f"\n[TEXT TOKENS - Pre-prompt]")
    print(f"{sample['pre_prompt']}")

    # Time series text + embeddings
    for i, ts_text in enumerate(sample['time_series_text']):
        ts_length = len(sample['time_series'][i])
        num_patches = ts_length // PATCH_SIZE
        print(f"\n[TEXT TOKENS - Time Series Description {i+1}]")
        print(f"{ts_text}")
        print(f"\n[TIME SERIES EMBEDDINGS - {num_patches} embedding tokens inserted here]")
        print(f"   ↳ Raw time series: {ts_length} data points")
        print(f"   ↳ Patch size: {PATCH_SIZE}")
        print(f"   ↳ Number of patches: {num_patches}")
        print(f"   ↳ Each patch → 1 embedding token (2048-dimensional)")
        print(f"   ↳ Model can attend to all {num_patches} time series tokens")

    # Post-prompt
    print(f"\n[TEXT TOKENS - Post-prompt/Question]")
    print(f"{sample['post_prompt']}")

    print(f"\n{'─'*100}")

    # Answers
    expected = sample['answer'].replace(model.get_eos_token(), '').strip()
    generated = prediction.replace(model.get_eos_token(), '').strip()

    # Extract just the answer part from generated text
    # The model might repeat the prompt, so try to extract just the answer
    if expected in generated:
        generated_answer = expected  # Model got it right
    else:
        # Try to extract after common patterns
        for split_word in ['Answer:', 'answer:', expected[:3]]:
            if split_word in generated:
                parts = generated.split(split_word)
                if len(parts) > 1:
                    generated_answer = parts[-1].strip().split()[0] if parts[-1].strip() else generated
                    break
        else:
            generated_answer = generated.strip().split()[0] if generated.strip() else generated

    print(f"\n🎯 EXPECTED ANSWER (Ground Truth):")
    print(f"   {expected}")

    print(f"\n🤖 MODEL'S GENERATED ANSWER:")
    print(f"   {generated_answer}")

    # Check correctness
    is_correct = expected.lower() in generated.lower() or generated_answer.lower() == expected.lower()
    if is_correct:
        print(f"\n✅ CORRECT!")
    else:
        print(f"\n❌ INCORRECT")

    print(f"\n{'='*100}\n")

    return is_correct


# Training loop
print(f"\n4. Starting training...")
print(f"{'='*100}\n")

best_val_loss = float('inf')

for epoch in range(1, NUM_EPOCHS + 1):
    # Training
    model.train()
    running_loss = 0.0

    prog = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    for batch in prog:
        optimizer.zero_grad()
        loss = model.compute_loss(batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        prog.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = running_loss / len(train_loader)
    print(f"\n📈 Epoch {epoch} - Train Loss: {avg_train_loss:.4f}")

    # Evaluation on test set with detailed logging
    if epoch == NUM_EPOCHS:  # Only log on final epoch
        print(f"\n{'='*100}")
        print(f"FINAL EVALUATION - DETAILED LOGGING")
        print(f"{'='*100}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for idx, batch in enumerate(test_loader):
                # Generate prediction
                predictions = model.generate(batch, max_new_tokens=50)

                # Log details for each sample
                for sample, pred in zip(batch, predictions):
                    is_correct = log_sample_with_details(sample, pred, idx)
                    if is_correct:
                        correct += 1
                    total += 1

                # Only log first 5 samples
                if idx >= 4:
                    print(f"\n... (showing first 5 samples, continuing evaluation)")
                    break

        # Continue evaluation for remaining samples (without logging)
        if idx < len(test_loader) - 1:
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx <= idx:
                    continue
                with torch.no_grad():
                    predictions = model.generate(batch, max_new_tokens=50)
                    for sample, pred in zip(batch, predictions):
                        expected = sample['answer'].replace(model.get_eos_token(), '').strip()
                        is_correct = expected.lower() in pred.lower()
                        if is_correct:
                            correct += 1
                        total += 1

        accuracy = correct / total if total > 0 else 0
        print(f"\n{'='*100}")
        print(f"FINAL RESULTS")
        print(f"{'='*100}")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total} correct)")
        print(f"{'='*100}\n")

print(f"\n✅ Training completed!")
print(f"Check the logs above to see:")
print(f"  1. Full prompt structure with time series embedding markers")
print(f"  2. Model's generated answers")
print(f"  3. Expected ground truth answers")
