#!/usr/bin/env python3
"""
Test script for untrained encoder on Polymarket trend prediction.
Logs sample questions and answers to evaluate output format following.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
import re

from time_series_datasets.polymarket.PolymarketTrendDataset import PolymarketTrendDataset
from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from model.llm.OpenTSLMSP import OpenTSLMSP
from model_config import PATCH_SIZE


def extract_answer(text: str) -> str:
    """
    Extract 'increasing' or 'decreasing' from model output (fuzzy matching).

    Args:
        text: Model output text

    Returns:
        'increasing', 'decreasing', or 'unknown'
    """
    text_lower = text.lower()

    # Look for the words anywhere in the output
    has_increasing = 'increasing' in text_lower or 'increase' in text_lower
    has_decreasing = 'decreasing' in text_lower or 'decrease' in text_lower

    # If both or neither appear, return unknown
    if has_increasing and has_decreasing:
        # Return the first one that appears
        inc_pos = text_lower.find('increas')
        dec_pos = text_lower.find('decreas')
        if inc_pos < dec_pos:
            return 'increasing'
        else:
            return 'decreasing'
    elif has_increasing:
        return 'increasing'
    elif has_decreasing:
        return 'decreasing'
    else:
        return 'unknown'


def test_untrained_encoder(
    num_samples: int = 20,
    device: str = "cuda",
    output_dir: str = "/local/home/wangni/results/untrained_test"
):
    """
    Test untrained encoder on Polymarket dataset.

    Args:
        num_samples: Number of samples to test
        device: Device to use
        output_dir: Directory to save results
    """
    print("="*80)
    print("Testing Untrained Encoder on Polymarket Trend Prediction")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model with Gemma 270M (untrained encoder)
    print(f"\n[1/4] Initializing model on {device}...")
    model = OpenTSLMSP(llm_id="google/gemma-3-270m", device=device).to(device)
    model.eval()
    print("✓ Model initialized (encoder and projector are randomly initialized)")

    # Load test dataset
    print(f"\n[2/4] Loading test dataset...")
    try:
        test_dataset = PolymarketTrendDataset("test", model.get_eos_token())
        print(f"✓ Loaded {len(test_dataset)} test samples")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run the data processing script first:")
        print("  cd /data/local/home/wangni/OpenTSLM")
        print("  conda run -n tslm python -m src.time_series_datasets.polymarket.PolymarketTrendDataset")
        return

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    # Test on samples
    print(f"\n[3/4] Running inference on {num_samples} samples...")
    results = []
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing", total=num_samples)):
            if idx >= num_samples:
                break

            # Generate prediction
            predictions = model.generate(batch, max_new_tokens=50)

            for sample, pred in zip(batch, predictions):
                # Extract information
                question_type = sample.get('question_type', 'unknown')
                ground_truth = sample.get('answer', '').strip()
                market_question = sample.get('question_text', 'N/A')[:100]
                generated_text = pred.strip()

                # Extract answer from generated text (fuzzy matching)
                extracted_answer = extract_answer(generated_text)

                # Check correctness
                is_correct = (extracted_answer == ground_truth)
                if is_correct:
                    correct += 1
                total += 1

                # Store result
                result = {
                    'sample_idx': idx,
                    'question_type': question_type,
                    'market_question': market_question,
                    'full_prompt': sample.get('pre_prompt', '') + '\n' + sample.get('post_prompt', ''),
                    'generated_text': generated_text,
                    'extracted_answer': extracted_answer,
                    'ground_truth': ground_truth,
                    'correct': is_correct,
                }
                results.append(result)

                # Print first few samples
                if idx < 10:
                    print(f"\n{'='*80}")
                    print(f"Sample {idx + 1}")
                    print(f"{'='*80}")
                    print(f"Question Type: {question_type}")
                    print(f"Market: {market_question}")
                    print(f"\nModel Output:")
                    print(f"  {generated_text}")
                    print(f"\nExtracted Answer: {extracted_answer}")
                    print(f"Ground Truth: {ground_truth}")
                    print(f"Correct: {'✓' if is_correct else '✗'}")

    # Calculate metrics
    print(f"\n[4/4] Results Summary:")
    print(f"{'='*80}")
    accuracy = correct / total if total > 0 else 0
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    # Count format following (did model output only "increasing" or "decreasing"?)
    format_following = sum(
        1 for r in results
        if r['generated_text'].strip().lower() in ['increasing', 'decreasing']
    )
    format_rate = format_following / total if total > 0 else 0
    print(f"\nFormat Following: {format_following}/{total} ({format_rate:.2%})")

    # Breakdown by question type
    past_results = [r for r in results if r['question_type'] == 'past_trend']
    future_results = [r for r in results if r['question_type'] == 'future_trend']

    if past_results:
        past_acc = sum(1 for r in past_results if r['correct']) / len(past_results)
        print(f"\nPast Trend Accuracy: {past_acc:.2%} ({len(past_results)} samples)")

    if future_results:
        future_acc = sum(1 for r in future_results if r['correct']) / len(future_results)
        print(f"Future Trend Accuracy: {future_acc:.2%} ({len(future_results)} samples)")

    # Save detailed results
    results_file = os.path.join(output_dir, "untrained_encoder_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'summary': {
                'total_samples': total,
                'correct': correct,
                'accuracy': accuracy,
                'format_following': format_following,
                'format_rate': format_rate,
            },
            'results': results,
        }, f, indent=2)

    print(f"\n✓ Detailed results saved to: {results_file}")

    # Save sample outputs to text file for easy reading
    samples_file = os.path.join(output_dir, "untrained_encoder_samples.txt")
    with open(samples_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("UNTRAINED ENCODER TEST SAMPLES\n")
        f.write("="*80 + "\n\n")

        for r in results[:20]:  # Save first 20 samples
            f.write(f"Sample {r['sample_idx'] + 1}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Question Type: {r['question_type']}\n")
            f.write(f"Market: {r['market_question']}\n")
            f.write(f"\nModel Output:\n{r['generated_text']}\n")
            f.write(f"\nExtracted: {r['extracted_answer']}\n")
            f.write(f"Ground Truth: {r['ground_truth']}\n")
            f.write(f"Correct: {'✓' if r['correct'] else '✗'}\n")
            f.write("\n" + "="*80 + "\n\n")

    print(f"✓ Sample outputs saved to: {samples_file}")

    print("\n" + "="*80)
    print("✓ Untrained Encoder Test Complete!")
    print("="*80)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="/local/home/wangni/results/untrained_test", help="Output directory")

    args = parser.parse_args()

    test_untrained_encoder(
        num_samples=args.num_samples,
        device=args.device,
        output_dir=args.output_dir
    )
