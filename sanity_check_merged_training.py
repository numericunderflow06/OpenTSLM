#!/usr/bin/env python3
"""
Sanity Check for Merged Training
=================================
This script tests that the merged training pipeline works correctly with
datasets that have different time series shapes (different number of channels).

The check verifies:
1. Data loading from all 5 datasets
2. Batching with mixed channel counts (1, 3, 12 channels)
3. Forward pass through the model
4. Loss computation
5. Backward pass (gradient computation)

If all checks pass, the merged training should work correctly.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from time_series_datasets.TSQADataset import TSQADataset
from time_series_datasets.m4.M4QADataset import M4QADataset
from time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from model_config import PATCH_SIZE


def run_sanity_check(
    device: str = "cuda",
    llm_id: str = "meta-llama/Llama-3.2-1B",
    samples_per_dataset: int = 10,
    batch_size: int = 4,
    num_training_steps: int = 3,
):
    """
    Run a sanity check on the merged training pipeline.

    Args:
        device: Device to use ('cuda', 'cpu')
        llm_id: LLM model ID
        samples_per_dataset: Number of samples to load from each dataset
        batch_size: Batch size for testing
        num_training_steps: Number of training steps to run
    """
    print("=" * 70)
    print("Merged Training Sanity Check")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"LLM: {llm_id}")
    print(f"Samples per dataset: {samples_per_dataset}")
    print(f"Batch size: {batch_size}")
    print(f"Training steps: {num_training_steps}")
    print()

    # =========================================================================
    # Step 1: Load datasets
    # =========================================================================
    print("[Step 1/6] Loading datasets...")

    dataset_classes = [
        ("TSQA", TSQADataset),
        ("M4QA", M4QADataset),
        ("HARCoT", HARCoTQADataset),
        ("SleepEDF", SleepEDFCoTQADataset),
        ("ECG-QA", ECGQACoTQADataset),
    ]

    # Initialize model to get EOS token
    print(f"  Initializing {llm_id}...")
    model = OpenTSLMFlamingo(
        device=device,
        llm_id=llm_id,
        cross_attn_every_n_layers=1,
    ).to(device)  # Ensure all model components are on the correct device
    eos_token = model.get_eos_token()
    print(f"  EOS token: {repr(eos_token)}")

    datasets = []
    for name, ds_class in dataset_classes:
        print(f"  Loading {name}...", end=" ", flush=True)
        try:
            ds = ds_class("train", EOS_TOKEN=eos_token)
            # Limit samples
            ds.dataset = ds.dataset[:samples_per_dataset]
            datasets.append(ds)
            print(f"OK ({len(ds)} samples)")
        except Exception as e:
            print(f"FAILED: {e}")
            return False

    print(f"  Total samples: {sum(len(ds) for ds in datasets)}")
    print()

    # =========================================================================
    # Step 2: Check time series shapes
    # =========================================================================
    print("[Step 2/6] Checking time series shapes...")

    shapes_info = []
    for ds, (name, _) in zip(datasets, dataset_classes):
        sample = ds[0]
        ts = sample["time_series"]
        # Convert to tensor if needed
        if isinstance(ts, list):
            ts = [torch.as_tensor(t, dtype=torch.float32) for t in ts]
            num_series = len(ts)
            if num_series > 0:
                shape = ts[0].shape
            else:
                shape = "empty"
        else:
            num_series = ts.shape[0] if len(ts.shape) > 1 else 1
            shape = ts.shape
        shapes_info.append((name, num_series, shape))
        print(f"  {name}: {num_series} series, shape={shape}")

    # Verify different channel counts exist
    channel_counts = set()
    for name, num, shape in shapes_info:
        if isinstance(shape, tuple) and len(shape) >= 1:
            channel_counts.add(shape[0] if len(shape) == 1 else num)

    if len(channel_counts) > 1:
        print(f"  [OK] Found {len(channel_counts)} different channel counts: {channel_counts}")
    else:
        print(f"  [WARN] All datasets have the same shape - may not fully test channel padding")
    print()

    # =========================================================================
    # Step 3: Create merged dataloader
    # =========================================================================
    print("[Step 3/6] Creating merged dataloader...")

    merged_ds = ConcatDataset(datasets)
    dataloader = DataLoader(
        merged_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    print(f"  Merged dataset size: {len(merged_ds)}")
    print(f"  Number of batches: {len(dataloader)}")
    print()

    # =========================================================================
    # Step 4: Test forward pass
    # =========================================================================
    print("[Step 4/6] Testing forward pass...")

    model.eval()
    batch_shapes = []

    try:
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= 2:  # Test first 2 batches
                    break

                # Log batch info
                ts_shapes = [item["time_series"].shape for item in batch]
                print(f"  Batch {i+1}: {len(batch)} samples, TS shapes: {ts_shapes}")
                batch_shapes.extend(ts_shapes)

                # Forward pass (generate)
                predictions = model.generate(batch, max_new_tokens=10)
                print(f"    Generated {len(predictions)} predictions")
                print(f"    Sample prediction: {predictions[0][:50]}...")

        print("  [OK] Forward pass successful")
    except Exception as e:
        print(f"  [FAILED] Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # =========================================================================
    # Step 5: Test loss computation
    # =========================================================================
    print("[Step 5/6] Testing loss computation...")

    model.train()

    try:
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Test first 2 batches
                break

            loss = model.compute_loss(batch)
            print(f"  Batch {i+1}: loss = {loss.item():.4f}")

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [FAILED] Loss is NaN or Inf")
                return False

        print("  [OK] Loss computation successful")
    except Exception as e:
        print(f"  [FAILED] Loss computation error: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # =========================================================================
    # Step 6: Test backward pass (training step)
    # =========================================================================
    print("[Step 6/6] Testing training steps...")

    # Get parameters to optimize
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-5)

    print(f"  Trainable parameters: {sum(p.numel() for p in params_to_optimize):,}")

    try:
        model.train()
        losses = []

        for step, batch in enumerate(dataloader):
            if step >= num_training_steps:
                break

            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()

            # Check gradients
            grad_norms = []
            for p in params_to_optimize[:5]:  # Check first 5 params
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())

            optimizer.step()
            losses.append(loss.item())

            print(f"  Step {step+1}: loss = {loss.item():.4f}, grad_norms = {grad_norms[:3]}")

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [FAILED] Loss became NaN or Inf")
                return False

        print("  [OK] Training steps successful")
        print(f"  Loss progression: {' -> '.join(f'{l:.4f}' for l in losses)}")
    except Exception as e:
        print(f"  [FAILED] Training step error: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("SANITY CHECK PASSED!")
    print("=" * 70)
    print()
    print("The merged training pipeline is working correctly:")
    print("  - All 5 datasets loaded successfully")
    print("  - Batching with different channel counts works")
    print("  - Forward pass (generation) works")
    print("  - Loss computation works")
    print("  - Backward pass (training) works")
    print()
    print("You can now run the full merged training with confidence.")
    print()

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity check for merged training")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM model ID",
    )
    parser.add_argument(
        "--samples_per_dataset",
        type=int,
        default=10,
        help="Number of samples per dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=3,
        help="Number of training steps to test",
    )

    args = parser.parse_args()

    success = run_sanity_check(
        device=args.device,
        llm_id=args.llm_id,
        samples_per_dataset=args.samples_per_dataset,
        batch_size=args.batch_size,
        num_training_steps=args.num_training_steps,
    )

    sys.exit(0 if success else 1)
