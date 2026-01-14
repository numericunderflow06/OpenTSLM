#!/usr/bin/env python3
"""
Sanity Check Script for Merged Dataset Training
------------------------------------------------
This script runs a quick sanity check with 1 epoch of training,
validation, and evaluation to catch any potential code bugs before
launching the full training run.

Usage:
    python sanity_check.py --model OpenTSLMFlamingo

This will:
1. Load a small subset of data (10 samples per dataset)
2. Run 1 epoch of training
3. Validate on the subset
4. Evaluate on the test subset
5. Report any errors encountered
"""

import sys
import os
import time
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import argparse
import torch

from merged_training import MergedTrainer


def run_sanity_check(
    model_type: str = "OpenTSLMFlamingo",
    llm_id: str = "meta-llama/Llama-3.2-1B",
    device: str = None,
    max_samples: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Run a sanity check with minimal data and 1 epoch.

    Args:
        model_type: Model type ('OpenTSLMSP' or 'OpenTSLMFlamingo')
        llm_id: LLM model ID
        device: Device to use (None for auto-detect)
        max_samples: Max samples per dataset for sanity check
        verbose: Print verbose output

    Returns:
        Dictionary with sanity check results
    """
    results = {
        "passed": False,
        "model_type": model_type,
        "llm_id": llm_id,
        "errors": [],
        "warnings": [],
        "timings": {},
    }

    print("=" * 60)
    print("SANITY CHECK - Merged Dataset Training")
    print("=" * 60)
    print(f"Model: {model_type}")
    print(f"LLM: {llm_id}")
    print(f"Max samples per dataset: {max_samples}")
    print(f"Device: {device or 'auto'}")
    print("=" * 60)
    print()

    try:
        # Step 1: Initialize trainer
        print("[1/4] Initializing trainer...")
        start_time = time.time()

        trainer = MergedTrainer(
            model_type=model_type,
            device=device,
            gradient_checkpointing=False,
            llm_id=llm_id,
        )

        results["timings"]["init"] = time.time() - start_time
        print(f"      Trainer initialized in {results['timings']['init']:.1f}s")
        print(f"      Device: {trainer.device}")
        print()

        # Step 2: Run training with 1 epoch
        print("[2/4] Running 1 epoch training + validation + evaluation...")
        start_time = time.time()

        training_results = trainer.train_merged(
            num_epochs=1,
            batch_size=2,  # Small batch for sanity check
            max_samples_per_dataset=max_samples,
            eval_only=False,
        )

        results["timings"]["training"] = time.time() - start_time
        print(f"      Training completed in {results['timings']['training']:.1f}s")
        print()

        # Step 3: Verify results
        print("[3/4] Verifying results...")

        # Check that we got valid results
        if training_results is None:
            results["errors"].append("Training returned None")
        else:
            # Check for required keys
            required_keys = ["merged_test", "timeseriesexam1", "best_epoch", "best_val_loss"]
            for key in required_keys:
                if key not in training_results:
                    results["errors"].append(f"Missing key in results: {key}")

            # Check that best_val_loss is a valid number
            if "best_val_loss" in training_results:
                val_loss = training_results["best_val_loss"]
                if val_loss is None or (isinstance(val_loss, float) and (val_loss != val_loss)):  # NaN check
                    results["errors"].append(f"Invalid validation loss: {val_loss}")
                elif val_loss == float("inf"):
                    results["warnings"].append("Validation loss is infinity (no improvement)")
                else:
                    print(f"      Best validation loss: {val_loss:.4f}")

            # Check TimeSeriesExam1 results
            if "timeseriesexam1" in training_results:
                tsexam = training_results["timeseriesexam1"]
                if "accuracy" in tsexam:
                    print(f"      TimeSeriesExam1 accuracy: {tsexam['accuracy']:.4f}")

        print()

        # Step 4: Check for output files
        print("[4/4] Checking output files...")
        results_dir = trainer.results_dir

        expected_files = [
            os.path.join(results_dir, "checkpoints", "best_model.pt"),
            os.path.join(results_dir, "checkpoints", "loss_history.txt"),
        ]

        for filepath in expected_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                print(f"      Found: {os.path.basename(filepath)} ({size:,} bytes)")
            else:
                results["warnings"].append(f"Expected file not found: {filepath}")

        print()

        # Final verdict
        if not results["errors"]:
            results["passed"] = True
            print("=" * 60)
            print("SANITY CHECK PASSED")
            print("=" * 60)
        else:
            print("=" * 60)
            print("SANITY CHECK FAILED")
            print("=" * 60)
            print("Errors:")
            for error in results["errors"]:
                print(f"  - {error}")

        if results["warnings"]:
            print("\nWarnings:")
            for warning in results["warnings"]:
                print(f"  - {warning}")

        results["training_results"] = training_results

    except Exception as e:
        results["errors"].append(f"Exception: {str(e)}")
        results["traceback"] = traceback.format_exc()

        print("=" * 60)
        print("SANITY CHECK FAILED WITH EXCEPTION")
        print("=" * 60)
        print(f"Error: {e}")
        print()
        print("Traceback:")
        print(results["traceback"])

    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Sanity Check for Merged Dataset Training")
    parser.add_argument(
        "--model",
        type=str,
        choices=["OpenTSLMSP", "OpenTSLMFlamingo"],
        default="OpenTSLMFlamingo",
        help="Model type to test (default: OpenTSLMFlamingo)",
    )
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM model ID (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Max samples per dataset (default: 10)",
    )

    args = parser.parse_args()

    results = run_sanity_check(
        model_type=args.model,
        llm_id=args.llm_id,
        device=args.device,
        max_samples=args.max_samples,
    )

    # Exit with appropriate code
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
