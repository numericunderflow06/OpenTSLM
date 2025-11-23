#!/usr/bin/env python3
"""
Zero-shot evaluation of llama-3.2-1b-ecg-flamingo on MIMIC-IV ECG-QA mini-100 dataset.

This script loads the pretrained Flamingo model from HuggingFace and evaluates it on
our 100-question MIMIC-IV subset, logging all results including exact outputs,
ground truth, and accuracies.

FIXES:
- Uses max_patches=1024 to match the checkpoint (was causing size mismatch error)
- Comprehensive logging to file and console
- All outputs saved with detailed metrics
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download


def setup_logging(results_dir):
    """Setup logging to both file and console."""
    log_file = results_dir / "evaluation.log"

    # Create logger
    logger = logging.getLogger('eval')
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def build_ecg_id_to_path_mapping(ecg_data_root, logger):
    """Build mapping from study ID to file path using a single efficient scan."""
    import subprocess

    logger.info("Building ECG ID to path mapping (using single optimized scan)...")

    # Use find command to get all .dat files in one go
    try:
        result = subprocess.run(
            ['find', ecg_data_root, '-name', '*.dat', '-type', 'f'],
            capture_output=True,
            text=True,
            timeout=60
        )

        ecg_id_to_path = {}
        for path in result.stdout.strip().split('\n'):
            if not path:
                continue
            # Extract study ID from path like: .../s12345678/12345678.dat
            parts = path.split('/')
            for i, part in enumerate(parts):
                if part.startswith('s') and i + 1 < len(parts):
                    filename = parts[i + 1]
                    if filename.endswith('.dat'):
                        study_id_str = filename[:-4]  # Remove .dat
                        try:
                            study_id = int(study_id_str)
                            ecg_id_to_path[study_id] = path
                            break
                        except ValueError:
                            continue

        logger.info(f"Built mapping for {len(ecg_id_to_path)} ECG files")
        return ecg_id_to_path

    except subprocess.TimeoutExpired:
        logger.error("Timeout while scanning ECG directory")
        return {}
    except Exception as e:
        logger.error(f"Error building ECG mapping: {e}")
        return {}


def load_mini100_dataset(dataset_path, ecg_data_path, eos_token, logger):
    """Load the MIMIC-IV mini-100 dataset."""
    import json
    from datasets import Dataset

    logger.info("Loading mini-100 dataset...")

    # Load the train split (which contains all 100 QA pairs)
    qa_samples = []
    qa_file = os.path.join(dataset_path, "paraphrased/train/000000.json")

    with open(qa_file, 'r') as f:
        qa_samples.extend(json.load(f))

    logger.info(f"Loaded {len(qa_samples)} QA pairs from {qa_file}")

    # Load ECG IDs
    ecg_ids = []
    with open(os.path.join(dataset_path, "train_ecgs.tsv"), 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                ecg_ids.append(int(parts[1]))

    logger.info(f"Loaded {len(ecg_ids)} unique ECG IDs")

    # Build a mapping from ECG IDs to paths (single efficient scan)
    ecg_id_to_path = build_ecg_id_to_path_mapping(ecg_data_path, logger)

    # Check how many of our needed ECGs were found
    found_count = sum(1 for ecg_id in ecg_ids if ecg_id in ecg_id_to_path)
    logger.info(f"Found paths for {found_count}/{len(ecg_ids)} needed ECG files")

    for ecg_id in ecg_ids:
        if ecg_id not in ecg_id_to_path:
            logger.warning(f"Could not find ECG file for study ID: {ecg_id}")

    # Convert answer from list to string and add required fields for each sample
    for sample in qa_samples:
        if isinstance(sample.get('answer'), list):
            # Convert list to string (take first element for single answers)
            sample['answer'] = sample['answer'][0] if len(sample['answer']) > 0 else ""

        # Add clinical_contexts if missing (required by ECGQAMimicIVDataset)
        if 'clinical_contexts' not in sample:
            sample['clinical_contexts'] = [""]  # Empty context for zero-shot evaluation

        # Add ecg_paths based on ecg_id
        if 'ecg_paths' not in sample and 'ecg_id' in sample:
            ecg_ids_list = sample['ecg_id'] if isinstance(sample['ecg_id'], list) else [sample['ecg_id']]
            ecg_paths_list = []
            for eid in ecg_ids_list:
                if eid in ecg_id_to_path:
                    ecg_paths_list.append(ecg_id_to_path[eid])
            sample['ecg_paths'] = ecg_paths_list

    # Convert to Dataset
    dataset = Dataset.from_list(qa_samples)

    # Import and temporarily configure the loader
    import time_series_datasets.ecg_qa.mimiciv_ecg_loader as loader_module

    # Store original loader
    original_loader = loader_module.load_ecg_qa_mimiciv_splits
    original_ecg_path = getattr(loader_module, 'MIMIC_IV_ECG_DATA_DIR', None)

    # Set custom paths
    loader_module.MIMIC_IV_ECG_DATA_DIR = ecg_data_path
    logger.debug(f"Set MIMIC_IV_ECG_DATA_DIR to: {ecg_data_path}")

    def load_custom_mini100():
        return dataset, dataset, dataset

    loader_module.load_ecg_qa_mimiciv_splits = load_custom_mini100

    try:
        # Import dataset class
        from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset

        # Create dataset
        ds = ECGQAMimicIVDataset(
            split="train",
            EOS_TOKEN=eos_token,
            use_cot_format=False,  # Use simple format for zero-shot
            exclude_comparison=False,
            preload_processed_data=False,  # Don't preload to save memory
        )

        logger.info(f"Created ECGQAMimicIVDataset with {len(ds)} samples")
        return ds

    finally:
        # Restore original loader
        loader_module.load_ecg_qa_mimiciv_splits = original_loader
        if original_ecg_path is not None:
            loader_module.MIMIC_IV_ECG_DATA_DIR = original_ecg_path


def normalize_answer(answer):
    """Normalize answer for comparison."""
    if isinstance(answer, list):
        answer = answer[0] if len(answer) > 0 else ""
    return str(answer).lower().strip()


def check_answer_correctness(prediction, ground_truth):
    """Check if prediction matches ground truth."""
    pred_normalized = normalize_answer(prediction)
    gt_normalized = normalize_answer(ground_truth)

    # Check if ground truth appears in prediction or vice versa
    return gt_normalized in pred_normalized or pred_normalized == gt_normalized


def create_modified_opentslm_flamingo(device, llm_id, cross_attn_every_n_layers, max_patches=1024):
    """
    Create OpenTSLMFlamingo with custom max_patches to match checkpoint.

    This is necessary because the checkpoint was trained with max_patches=1024,
    but the default OpenTSLMFlamingo uses max_patches=2600.
    """
    from types import SimpleNamespace
    from model.encoder.CNNTokenizer import CNNTokenizer
    from model.llm.TimeSeriesFlamingoWithTrainableEncoder import TimeSeriesFlamingoWithTrainableEncoder
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from open_flamingo.open_flamingo.src.flamingo_lm import FlamingoLMMixin
    from open_flamingo.open_flamingo.src.utils import extend_instance
    from model_config import ENCODER_OUTPUT_DIM

    # Create encoder with custom max_patches
    time_series_encoder = CNNTokenizer(max_patches=max_patches).to(device)

    # Create tokenizer
    cache_dir = "/local/home/wangni/.cache/huggingface"
    text_tokenizer = AutoTokenizer.from_pretrained(
        llm_id,
        local_files_only=False,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # Load language model
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        llm_id,
        local_files_only=False,
        trust_remote_code=True,
        device_map={"": device},
        attn_implementation="eager",
        cache_dir=cache_dir,
    )

    # Add Flamingo special tokens
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Convert to Flamingo
    extend_instance(lang_encoder, FlamingoLMMixin)

    # Infer decoder layers
    def _infer_decoder_layers_attr_name(model):
        model_class_name = model.__class__.__name__.lower()
        if "llama" in model_class_name:
            return "model.layers"
        elif "opt" in model_class_name:
            return "model.decoder.layers"
        elif "gpt" in model_class_name:
            return "transformer.h"
        else:
            # Default for most causal LMs
            return "model.layers"

    decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    # Initialize Flamingo with trainable encoder
    # Wrap encoder in SimpleNamespace as expected by Flamingo
    flamingo_model = TimeSeriesFlamingoWithTrainableEncoder(
        vision_encoder=SimpleNamespace(visual=time_series_encoder),
        lang_encoder=lang_encoder,
        eoc_token_id=text_tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1],
        media_token_id=text_tokenizer("<image>", add_special_tokens=False)["input_ids"][-1],
        vis_dim=ENCODER_OUTPUT_DIM,  # 128
        cross_attn_every_n_layers=cross_attn_every_n_layers,
    )

    # Return the raw flamingo model
    # We'll handle data preprocessing in the evaluation loop using OpenTSLMFlamingo's logic as reference
    return flamingo_model, text_tokenizer


def evaluate_model_on_mimiciv_mini100():
    """Evaluate the model on the MIMIC-IV ECG-QA mini-100 dataset."""

    # Setup
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results_evaluation/llama-3.2-1b-ecg-flamingo_mimiciv_mini100_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir)
    logger.info("="*60)
    logger.info("MIMIC-IV ECG-QA Mini-100 Evaluation")
    logger.info("="*60)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Timestamp: {timestamp}")

    # Load model
    logger.info("\n" + "="*60)
    logger.info("Loading model: OpenTSLM/llama-3.2-1b-ecg-flamingo")
    logger.info("="*60)

    try:
        llm_id = "meta-llama/Llama-3.2-1B"
        max_patches = 1024  # CRITICAL: Must match checkpoint

        logger.info(f"Base LLM: {llm_id}")
        logger.info(f"Max patches: {max_patches} (matches checkpoint)")
        logger.info("Creating Flamingo model...")

        # Create model with correct max_patches
        flamingo_model, text_tokenizer = create_modified_opentslm_flamingo(
            device=device,
            llm_id=llm_id,
            cross_attn_every_n_layers=1,
            max_patches=max_patches,
        )
        flamingo_model.to(device)

        logger.info("✓ Model structure created")

        # Download and load checkpoint
        logger.info("Downloading checkpoint from HuggingFace...")
        checkpoint_path = hf_hub_download(
            repo_id="OpenTSLM/llama-3.2-1b-ecg-flamingo",
            filename="softprompt-llama_3_2_1b-ecg.pt",
            cache_dir="/local/home/wangni/.cache/huggingface"
        )

        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load model state
        if 'model_state' in ckpt:
            # Fix: Checkpoint has TWO copies of weights with different prefixes:
            #   - 'model.*' = Flamingo model weights (716 keys)
            #   - 'llm.*' = Duplicate LLM weights (716 keys)
            # We should ONLY use 'model.*' and strip the prefix
            state_dict = ckpt['model_state']

            # Only load keys starting with 'model.' and strip that prefix
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    # Strip 'model.' prefix
                    new_key = key[6:]  # Remove 'model.'
                    new_state_dict[new_key] = value
                # Ignore 'llm.*' keys - they're duplicates

            logger.info(f"Loaded {len(new_state_dict)} keys from checkpoint (ignored llm.* duplicates)")
            missing_keys, unexpected_keys = flamingo_model.load_state_dict(new_state_dict, strict=False)
            logger.info(f"✓ Loaded model from epoch {ckpt.get('epoch', '?')}")

            # Check for critical missing keys
            critical_prefixes = ['perceiver', 'gated_cross_attn_layers', 'vision_encoder']
            critical_missing = [k for k in missing_keys if any(k.startswith(p) for p in critical_prefixes)]

            if critical_missing:
                logger.error(f"✗ CRITICAL: Missing essential model components: {critical_missing[:10]}")
                logger.error("Cannot proceed with incomplete model. Exiting.")
                return

            if missing_keys:
                logger.warning(f"Non-critical missing keys ({len(missing_keys)} total): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"Unexpected keys ({len(unexpected_keys)} total): {unexpected_keys[:5]}...")
            elif missing_keys:
                logger.warning(f"Missing {len(missing_keys)} non-critical keys (first 10): {missing_keys[:10]}")
                
            if unexpected_keys:
                logger.warning(f"Unexpected {len(unexpected_keys)} keys (first 10): {unexpected_keys[:10]}")
        else:
            logger.error("✗ Checkpoint format not recognized")
            return

        flamingo_model.eval()
        logger.info("✓ Model loaded and set to eval mode")

        # Store for generation
        model_info = {
            'model': flamingo_model,
            'tokenizer': text_tokenizer,
            'device': device,
        }

    except Exception as e:
        logger.error(f"✗ Error loading model: {e}", exc_info=True)
        return

    # Load mini-100 dataset
    logger.info("\n" + "="*60)
    logger.info("Loading MIMIC-IV ECG-QA mini-100 dataset")
    logger.info("="*60)

    dataset_path = "data/ecg-qa-mini-100"
    ecg_data_path = "data/mimic_iv_ecg/physionet.org"

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"ECG data path: {ecg_data_path}")

    try:
        dataset = load_mini100_dataset(dataset_path, ecg_data_path, text_tokenizer.eos_token, logger)
        logger.info(f"✓ Dataset loaded: {len(dataset)} samples")
    except Exception as e:
        logger.error(f"✗ Error loading dataset: {e}", exc_info=True)
        return

    # Create dataloader
    from time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=128
        ),
    )

    logger.info(f"Created dataloader with batch_size=1, patch_size=128")

    # Run evaluation
    logger.info("\n" + "="*60)
    logger.info("Running zero-shot evaluation")
    logger.info("="*60)

    # Create a wrapper that uses OpenTSLMFlamingo's data preprocessing logic
    class FlamingoWrapper:
        def __init__(self, llm, tokenizer, device):
            self.llm = llm
            self.text_tokenizer = tokenizer
            self.device = device

        def generate(self, batch, max_new_tokens=50):
            """Generate using proper data preprocessing from OpenTSLMFlamingo."""
            # Import the OpenTSLMFlamingo to use its pad_and_apply_batch method
            from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

            # Create a temporary instance just to use its data preprocessing
            # This is hacky but necessary since we can't instantiate OpenTSLMFlamingo with our loaded model
            temp_wrapper = type('TempWrapper', (), {
                'text_tokenizer': self.text_tokenizer,
                'device': self.device,
                'llm': self.llm
            })()

            # Use the pad_and_apply_batch method from OpenTSLMFlamingo
            input_ids, images, attention_mask, _ = OpenTSLMFlamingo.pad_and_apply_batch(
                temp_wrapper, batch, include_labels=True
            )

            # Generate
            with torch.inference_mode():
                gen_ids = self.llm.generate(
                    vision_x=images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    eos_token_id=self.text_tokenizer.eos_token_id,
                    pad_token_id=self.text_tokenizer.pad_token_id,
                )

                # Remove input ids from generation
                answer_only_ids = gen_ids[:, input_ids.shape[1]:]

                return self.text_tokenizer.batch_decode(answer_only_ids, skip_special_tokens=True)

    model_wrapper = FlamingoWrapper(flamingo_model, text_tokenizer, device)

    results = []
    correct = 0
    total = 0

    logger.info("Starting evaluation loop...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Generate prediction
            try:
                predictions = model_wrapper.generate(batch, max_new_tokens=100)
                prediction = predictions[0] if isinstance(predictions, list) else predictions
                
                # Debug logging for first sample
                if batch_idx == 0:
                    logger.info("=" * 60)
                    logger.info("DEBUG: First Sample Details")
                    logger.info("=" * 60)
                    logger.info(f"Question: {batch[0].get('question', 'N/A')[:200]}...")
                    logger.info(f"Pre-prompt length: {len(batch[0].get('pre_prompt', ''))}")
                    logger.info(f"Post-prompt length: {len(batch[0].get('post_prompt', ''))}")
                    logger.info(f"Time series count: {len(batch[0].get('time_series_text', []))}")
                    logger.info(f"Generated prediction: '{prediction}'")
                    logger.info(f"Prediction length: {len(prediction)}")
                    logger.info(f"Ground truth: '{batch[0].get('answer', 'N/A')}'")
                    logger.info("=" * 60)
                else:
                    logger.debug(f"Sample {batch_idx}: Generated prediction: {prediction[:100]}...")
            except Exception as e:
                logger.error(f"Error generating prediction for sample {batch_idx}: {e}")
                prediction = "[ERROR]"
                if batch_idx == 0:  # Print full traceback for first error
                    logger.error("Full traceback for first error:", exc_info=True)

            # Get ground truth
            sample = batch[0]
            ground_truth = sample.get("answer", "")

            # Check correctness
            is_correct = check_answer_correctness(prediction, ground_truth)
            if is_correct:
                correct += 1
            total += 1

            # Store result
            result = {
                "sample_id": batch_idx,
                "question": sample.get("question", ""),
                "question_type": sample.get("question_type", ""),
                "attribute_type": sample.get("attribute_type", ""),
                "ecg_id": sample.get("ecg_id", []),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "is_correct": is_correct,
            }
            results.append(result)

            # Print progress every 10 samples
            if (batch_idx + 1) % 10 == 0:
                current_acc = correct / total
                logger.info(f"Progress: {batch_idx + 1}/{len(dataset)} | Accuracy: {current_acc:.2%} ({correct}/{total})")
                logger.debug(f"Last prediction: {prediction[:100]}...")
                logger.debug(f"Ground truth: {ground_truth}")

    # Calculate final accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Calculate per-question-type accuracy
    question_type_stats = {}
    attribute_type_stats = {}

    for result in results:
        # Question type stats
        qtype = result["question_type"]
        if qtype not in question_type_stats:
            question_type_stats[qtype] = {"correct": 0, "total": 0}
        question_type_stats[qtype]["total"] += 1
        if result["is_correct"]:
            question_type_stats[qtype]["correct"] += 1

        # Attribute type stats
        atype = result["attribute_type"]
        if atype not in attribute_type_stats:
            attribute_type_stats[atype] = {"correct": 0, "total": 0}
        attribute_type_stats[atype]["total"] += 1
        if result["is_correct"]:
            attribute_type_stats[atype]["correct"] += 1

    for qtype in question_type_stats:
        stats = question_type_stats[qtype]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    for atype in attribute_type_stats:
        stats = attribute_type_stats[atype]
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"\nOverall Accuracy: {accuracy:.2%} ({correct}/{total})")

    logger.info(f"\nPer-Question-Type Accuracy:")
    for qtype, stats in sorted(question_type_stats.items()):
        logger.info(f"  {qtype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    logger.info(f"\nPer-Attribute-Type Accuracy:")
    for atype, stats in sorted(attribute_type_stats.items()):
        logger.info(f"  {atype}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    # Save results
    logger.info("\n" + "="*60)
    logger.info("Saving results")
    logger.info("="*60)

    # Save detailed results as JSONL
    results_file = results_dir / "detailed_results.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    logger.info(f"✓ Saved detailed results to: {results_file}")

    # Save summary metrics
    metrics = {
        "model": "OpenTSLM/llama-3.2-1b-ecg-flamingo",
        "base_llm": "meta-llama/Llama-3.2-1B",
        "checkpoint": "softprompt-llama_3_2_1b-ecg.pt",
        "architecture": "OpenTSLMFlamingo (TimeSeriesFlamingoWithTrainableEncoder)",
        "max_patches": 1024,
        "dataset": "MIMIC-IV ECG-QA mini-100",
        "dataset_path": dataset_path,
        "timestamp": timestamp,
        "device": device,
        "total_samples": total,
        "correct_predictions": correct,
        "overall_accuracy": accuracy,
        "question_type_accuracy": question_type_stats,
        "attribute_type_accuracy": attribute_type_stats,
    }

    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"✓ Saved metrics to: {metrics_file}")

    # Create a human-readable summary
    summary_file = results_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MIMIC-IV ECG-QA Mini-100 Evaluation Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model: OpenTSLM/llama-3.2-1b-ecg-flamingo\n")
        f.write(f"Base LLM: meta-llama/Llama-3.2-1B\n")
        f.write(f"Architecture: OpenTSLMFlamingo (TimeSeriesFlamingoWithTrainableEncoder)\n")
        f.write(f"Checkpoint: softprompt-llama_3_2_1b-ecg.pt\n")
        f.write(f"Max Patches: {max_patches}\n")
        f.write(f"Dataset: MIMIC-IV ECG-QA mini-100\n")
        f.write(f"Evaluation Date: {timestamp}\n")
        f.write(f"Device: {device}\n\n")
        f.write(f"Overall Results:\n")
        f.write(f"  Total Samples: {total}\n")
        f.write(f"  Correct: {correct}\n")
        f.write(f"  Accuracy: {accuracy:.2%}\n\n")
        f.write(f"Per-Question-Type Accuracy:\n")
        for qtype, stats in sorted(question_type_stats.items()):
            f.write(f"  {qtype}:\n")
            f.write(f"    Accuracy: {stats['accuracy']:.2%}\n")
            f.write(f"    Correct/Total: {stats['correct']}/{stats['total']}\n\n")
        f.write(f"Per-Attribute-Type Accuracy:\n")
        for atype, stats in sorted(attribute_type_stats.items()):
            f.write(f"  {atype}:\n")
            f.write(f"    Accuracy: {stats['accuracy']:.2%}\n")
            f.write(f"    Correct/Total: {stats['correct']}/{stats['total']}\n\n")

        # Add some example predictions
        f.write("\n" + "="*60 + "\n")
        f.write("Example Predictions (first 10 samples)\n")
        f.write("="*60 + "\n\n")
        for i, result in enumerate(results[:10]):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  Question: {result['question']}\n")
            f.write(f"  Ground Truth: {result['ground_truth']}\n")
            f.write(f"  Prediction: {result['prediction']}\n")
            f.write(f"  Correct: {'✓' if result['is_correct'] else '✗'}\n\n")

    logger.info(f"✓ Saved summary to: {summary_file}")

    logger.info("\n" + "="*60)
    logger.info("✅ Evaluation complete!")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {results_dir}")
    logger.info(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")

    return accuracy, results


if __name__ == "__main__":
    evaluate_model_on_mimiciv_mini100()
