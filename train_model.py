#!/usr/bin/env python3
"""
Training script for OpenTSLM encoder on Polymarket trend prediction.
Uses 4-GPU DDP training with wandb logging.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import json
import re
import wandb
import datetime

from time_series_datasets.polymarket.PolymarketTrendDataset import PolymarketTrendDataset
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from model.llm.OpenTSLMSP import OpenTSLMSP
from model_config import PATCH_SIZE

# Training configuration
BATCH_SIZE = 4
NUM_EPOCHS = 20
LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4
WEIGHT_DECAY = 1e-2
WARMUP_FRAC = 0.03
GRAD_CLIP_NORM = 1.0
EVAL_EVERY = 100  # Evaluate every N steps
SAVE_EVERY = 500  # Save checkpoint every N steps


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(hours=24)
        )
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def extract_answer(text: str) -> str:
    """Extract 'increasing' or 'decreasing' from model output."""
    text_lower = text.lower()

    has_increasing = 'increasing' in text_lower or 'increase' in text_lower
    has_decreasing = 'decreasing' in text_lower or 'decrease' in text_lower

    if has_increasing and has_decreasing:
        inc_pos = text_lower.find('increas')
        dec_pos = text_lower.find('decreas')
        return 'increasing' if inc_pos < dec_pos else 'decreasing'
    elif has_increasing:
        return 'increasing'
    elif has_decreasing:
        return 'decreasing'
    else:
        return 'unknown'


def evaluate_model(model, dataloader, device, rank, max_samples=50, global_step=0, save_samples=True):
    """Evaluate model on validation set and optionally save sample outputs."""
    model.eval()
    correct = 0
    total = 0
    sample_outputs = []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx >= max_samples:
                break

            try:
                predictions = model.generate(batch, max_new_tokens=50)

                for sample, pred in zip(batch, predictions):
                    ground_truth = sample.get('answer', '').strip()
                    extracted = extract_answer(pred)
                    is_correct = (extracted == ground_truth)

                    if is_correct:
                        correct += 1
                    total += 1

                    # Save first 10 samples for logging
                    if save_samples and len(sample_outputs) < 10:
                        sample_outputs.append({
                            'question_type': sample.get('question_type', 'unknown'),
                            'market': sample.get('question_text', 'N/A')[:80],
                            'generated_text': pred,
                            'extracted_answer': extracted,
                            'ground_truth': ground_truth,
                            'correct': is_correct
                        })

            except Exception as e:
                if rank == 0:
                    print(f"Warning: Evaluation error on batch {idx}: {e}")
                continue

    # Gather results across GPUs
    if dist.is_initialized():
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()

    # Save sample outputs to file (only rank 0)
    if rank == 0 and save_samples and sample_outputs:
        samples_dir = "/local/home/wangni/results/training_samples"
        os.makedirs(samples_dir, exist_ok=True)

        samples_file = os.path.join(samples_dir, f"samples_step_{global_step}.txt")
        with open(samples_file, 'w') as f:
            f.write(f"="*80 + "\n")
            f.write(f"Training Step {global_step} - Sample Outputs\n")
            f.write(f"="*80 + "\n\n")

            for i, s in enumerate(sample_outputs, 1):
                f.write(f"Sample {i}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"Type: {s['question_type']}\n")
                f.write(f"Market: {s['market']}\n")
                f.write(f"\nGenerated: {s['generated_text']}\n")
                f.write(f"\nExtracted: {s['extracted_answer']}\n")
                f.write(f"Ground Truth: {s['ground_truth']}\n")
                f.write(f"Correct: {'✓' if s['correct'] else '✗'}\n")
                f.write(f"\n{'='*80}\n\n")

        print(f"  Sample outputs saved to: {samples_file}")

    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def train():
    """Main training function."""
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("="*80)
        print("OpenTSLM Training - Polymarket Trend Prediction")
        print("="*80)
        print(f"World size: {world_size}")
        print(f"Device: {device}")

    # Initialize wandb (only on rank 0)
    if rank == 0:
        wandb.init(
            project="opentslm-1101",
            config={
                "batch_size": BATCH_SIZE,
                "num_epochs": NUM_EPOCHS,
                "lr_encoder": LR_ENCODER,
                "lr_projector": LR_PROJECTOR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_frac": WARMUP_FRAC,
                "world_size": world_size,
                "model": "OpenTSLMSP",
                "llm": "Qwen/Qwen3-14B",
                "dataset": "polymarket_trend",
            }
        )

    # Initialize model
    if rank == 0:
        print("\n[1/6] Initializing model...")
        print("  Using 4-bit quantization to reduce memory usage...")

    model = OpenTSLMSP(llm_id="Qwen/Qwen3-14B", device=device, load_in_4bit=True).to(device)

    # Wrap with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        if rank == 0:
            print(f"✓ Model wrapped with DDP")

    # Get underlying model for parameter access
    model_module = model.module if hasattr(model, 'module') else model

    # Setup optimizer (only encoder and projector)
    if rank == 0:
        print("\n[2/6] Setting up optimizer...")

    enc_params = list(model_module.encoder.parameters())
    proj_params = list(model_module.projector.projector.parameters())

    optimizer = AdamW([
        {"params": enc_params, "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY},
        {"params": proj_params, "lr": LR_PROJECTOR, "weight_decay": WEIGHT_DECAY},
    ])

    if rank == 0:
        print(f"✓ Optimizer configured")
        print(f"  Encoder LR: {LR_ENCODER:.2e}")
        print(f"  Projector LR: {LR_PROJECTOR:.2e}")

    # Load datasets
    if rank == 0:
        print("\n[3/6] Loading datasets...")

    train_dataset = PolymarketTrendDataset("train", model_module.get_eos_token())
    val_dataset = PolymarketTrendDataset("validation", model_module.get_eos_token())

    if rank == 0:
        print(f"✓ Train: {len(train_dataset)} samples")
        print(f"✓ Val: {len(val_dataset)} samples")

    # Create dataloaders with distributed sampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    # Setup scheduler
    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_FRAC * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if rank == 0:
        print(f"\n[4/6] Training configuration:")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps}")
        print(f"  Steps per epoch: {len(train_loader)}")

    # Create checkpoint directory
    checkpoint_dir = "/local/home/wangni/results/polymarket_checkpoints"
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    if rank == 0:
        print(f"\n[5/6] Starting training...")
        print("="*80)

    global_step = 0
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        if train_sampler:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", disable=(rank != 0))

        for batch_idx, batch in enumerate(progress_bar):
            optimizer.zero_grad()

            # Forward pass
            loss = model_module.compute_loss(batch)

            # Backward pass
            loss.backward()
            clip_grad_norm_(model_module.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            # Track loss
            loss_val = loss.item()
            running_loss += loss_val
            epoch_loss += loss_val
            global_step += 1

            # Update progress bar
            if rank == 0:
                progress_bar.set_postfix({
                    'loss': f"{loss_val:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

            # Log to wandb
            if rank == 0 and global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss_val,
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': global_step,
                })

            # Evaluate periodically
            if global_step % EVAL_EVERY == 0:
                if rank == 0:
                    print(f"\n[Evaluation at step {global_step}]")

                val_acc, val_correct, val_total = evaluate_model(
                    model_module, val_loader, device, rank, max_samples=50, global_step=global_step
                )

                if rank == 0:
                    print(f"  Val Accuracy: {val_acc:.2%} ({val_correct}/{val_total})")

                    wandb.log({
                        'val/accuracy': val_acc,
                        'val/correct': val_correct,
                        'val/total': val_total,
                        'train/step': global_step,
                    })

                    # Save best model
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
                        torch.save({
                            'epoch': epoch,
                            'step': global_step,
                            'encoder_state': model_module.encoder.state_dict(),
                            'projector_state': model_module.projector.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                            'val_acc': val_acc,
                        }, checkpoint_path)
                        print(f"  ✓ Saved best model (acc: {val_acc:.2%})")

                model.train()

            # Save checkpoint periodically
            if rank == 0 and global_step % SAVE_EVERY == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pt")
                torch.save({
                    'epoch': epoch,
                    'step': global_step,
                    'encoder_state': model_module.encoder.state_dict(),
                    'projector_state': model_module.projector.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }, checkpoint_path)

        # End of epoch summary
        avg_epoch_loss = epoch_loss / len(train_loader)

        if rank == 0:
            print(f"\n{'='*80}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Best Val Accuracy: {best_val_acc:.2%}")
            print(f"{'='*80}\n")

            wandb.log({
                'train/epoch_loss': avg_epoch_loss,
                'train/epoch': epoch,
            })

    # Final evaluation
    if rank == 0:
        print(f"\n[6/6] Final evaluation...")

    final_val_acc, final_correct, final_total = evaluate_model(
        model_module, val_loader, device, rank, max_samples=200, global_step=global_step, save_samples=True
    )

    if rank == 0:
        print(f"\n{'='*80}")
        print("Training Complete!")
        print(f"{'='*80}")
        print(f"Final Validation Accuracy: {final_val_acc:.2%} ({final_correct}/{final_total})")
        print(f"Best Validation Accuracy: {best_val_acc:.2%}")
        print(f"Checkpoints saved to: {checkpoint_dir}")

        wandb.log({
            'final/val_accuracy': final_val_acc,
            'final/best_val_accuracy': best_val_acc,
        })

        wandb.finish()

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    train()
