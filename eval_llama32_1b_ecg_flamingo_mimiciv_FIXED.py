#!/usr/bin/env python3
"""
Zero-shot evaluation of llama-3.2-1b-ecg-flamingo on MIMIC-IV ECG-QA mini-100 dataset.

FIXED VERSION - Uses proper ECGQAMimicIVDataset pipeline for correct data formatting!

Key fix: Instead of manually loading JSON and trying to format data ourselves, we now use
the ECGQAMimicIVDataset class which properly:
1. Loads and processes ECG signal files
2. Creates pre_prompt and post_prompt from questions
3. Formats time_series and time_series_text fields correctly
4. Applies proper downsampling and normalization
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
from typing import Tuple, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from datasets import Dataset
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


class Mini100ECGQADataset:
    """
    Custom dataset class that loads the mini-100 dataset using ECGQAMimicIVDataset's
    formatting pipeline to ensure proper data structure.
    """

    def __init__(self, dataset_path, ecg_data_path, eos_token, logger):
        """
        Load mini-100 dataset using ECGQAMimicIVDataset's processing pipeline.

        Args:
            dataset_path: Path to mini-100 dataset (data/ecg-qa-mini-100)
            ecg_data_path: Path to MIMIC-IV-ECG data
            eos_token: End-of-sequence token
            logger: Logger instance
        """
        from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset

        self.logger = logger
        self.eos_token = eos_token

        logger.info("Loading mini-100 dataset using ECGQAMimicIVDataset pipeline...")

        # Load raw JSON data
        qa_file = os.path.join(dataset_path, "paraphrased/train/000000.json")
        with open(qa_file, 'r') as f:
            qa_samples = json.load(f)

        logger.info(f"Loaded {len(qa_samples)} QA pairs from {qa_file}")

        # Load ECG IDs
        ecg_ids = []
        with open(os.path.join(dataset_path, "train_ecgs.tsv"), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    ecg_ids.append(int(parts[1]))

        logger.info(f"Loaded {len(ecg_ids)} unique ECG IDs")

        # Build ECG ID to path mapping
        ecg_id_to_path = self._build_ecg_id_to_path_mapping(ecg_data_path)
        logger.info(f"Found paths for {len(ecg_id_to_path)} ECG files")

        # Prepare samples with required fields for ECGQAMimicIVDataset
        for sample in qa_samples:
            # Convert answer from list to string
            if isinstance(sample.get('answer'), list):
                sample['answer'] = sample['answer'][0] if len(sample['answer']) > 0 else ""

            # Add clinical_contexts (empty for zero-shot)
            if 'clinical_contexts' not in sample:
                sample['clinical_contexts'] = [""]

            # Add ecg_paths based on ecg_id
            if 'ecg_paths' not in sample and 'ecg_id' in sample:
                ecg_ids_list = sample['ecg_id'] if isinstance(sample['ecg_id'], list) else [sample['ecg_id']]
                ecg_paths_list = []
                for eid in ecg_ids_list:
                    if eid in ecg_id_to_path:
                        ecg_paths_list.append(ecg_id_to_path[eid])
                    else:
                        logger.warning(f"Could not find ECG file for study ID: {eid}")
                sample['ecg_paths'] = ecg_paths_list

        # Create Dataset object
        raw_dataset = Dataset.from_list(qa_samples)
        logger.info(f"Created raw dataset with {len(raw_dataset)} samples")

        # Create a temporary ECGQAMimicIVDataset instance to use its formatting methods
        # We need to create a custom subclass that uses our mini-100 data

        class Mini100DatasetWrapper(ECGQAMimicIVDataset):
            """Wrapper that uses mini-100 data instead of loading from full dataset."""

            _mini100_data = None  # Class variable to store our data

            def _load_splits(self):
                """Override to use our mini-100 data instead of loading full dataset."""
                # Return our pre-loaded data for all splits
                # (We only use train split for evaluation)
                return self.__class__._mini100_data, Dataset.from_list([]), Dataset.from_list([])

        # Set the mini-100 data
        Mini100DatasetWrapper._mini100_data = raw_dataset

        # Create dataset instance - this will format all samples properly
        logger.info("Formatting samples using ECGQAMimicIVDataset pipeline...")
        formatted_dataset = Mini100DatasetWrapper(
            split="train",
            EOS_TOKEN=eos_token,
            format_sample_str=False,
            use_cot_format=False,  # Use simple format for zero-shot
            preload_processed_data=True  # Preload for better performance
        )

        self.dataset = formatted_dataset
        logger.info(f"Dataset ready with {len(self.dataset)} formatted samples")

        # Show example of formatted sample structure
        if len(self.dataset) > 0:
            sample = self.dataset[0]
            logger.info(f"Sample structure keys: {sample.keys()}")
            logger.info(f"Sample has time_series: {'time_series' in sample}")
            logger.info(f"Sample has time_series_text: {'time_series_text' in sample}")
            logger.info(f"Sample has pre_prompt: {'pre_prompt' in sample}")
            logger.info(f"Sample has post_prompt: {'post_prompt' in sample}")
            if 'time_series_text' in sample:
                logger.info(f"Number of time series: {len(sample['time_series_text'])}")

    def _build_ecg_id_to_path_mapping(self, ecg_data_root):
        """Build mapping from study ID to file path using efficient scan."""
        import subprocess

        self.logger.info("Building ECG ID to path mapping...")

        try:
            # Use find command to get all .dat files
            result = subprocess.run(
                ['find', ecg_data_root, '-name', '*.dat', '-type', 'f'],
                capture_output=True, text=True, timeout=60
            )

            ecg_id_to_path = {}
            for path in result.stdout.strip().split('\n'):
                if not path:
                    continue
                parts = path.split('/')
                for i, part in enumerate(parts):
                    if part.startswith('s') and i + 1 < len(parts):
                        filename = parts[i + 1]
                        if filename.endswith('.dat'):
                            study_id_str = filename[:-4]
                            try:
                                study_id = int(study_id_str)
                                ecg_id_to_path[study_id] = path
                                break
                            except ValueError:
                                continue

            return ecg_id_to_path

        except Exception as e:
            self.logger.error(f"Error building ECG mapping: {e}")
            return {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def create_modified_opentslm_flamingo(device, llm_id, cross_attn_every_n_layers, max_patches=1024):
    """
    Create OpenTSLMFlamingo with custom max_patches to match checkpoint.

    Args:
        device: Device to load model on
        llm_id: HuggingFace model ID for the base LLM
        cross_attn_every_n_layers: Cross-attention layer interval
        max_patches: Maximum number of patches (MUST be 1024 to match checkpoint)

    Returns:
        flamingo_model: The TimeSeriesFlamingoWithTrainableEncoder model
        text_tokenizer: The tokenizer for the LLM
    """
    from types import SimpleNamespace
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from model.encoder.CNNTokenizer import CNNTokenizer
    from model.llm.TimeSeriesFlamingoWithTrainableEncoder import TimeSeriesFlamingoWithTrainableEncoder
    from model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
    from open_flamingo.open_flamingo.src.utils import extend_instance
    from open_flamingo.open_flamingo.src.flamingo_lm import FlamingoLMMixin
    from model_config import ENCODER_OUTPUT_DIM

    # CRITICAL: Use max_patches=1024 to match checkpoint
    # (Default is 2600, which causes size mismatch)
    time_series_encoder = CNNTokenizer(max_patches=max_patches).to(device)

    # Load tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained(
        llm_id,
        local_files_only=False,
        trust_remote_code=True
    )

    # Load base LLM
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        llm_id,
        local_files_only=False,
        trust_remote_code=True,
        device_map={"": device},
        attn_implementation="eager",
    )

    # Add special tokens
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})

    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # CRITICAL: Extend LLM with Flamingo methods before creating Flamingo model
    extend_instance(lang_encoder, FlamingoLMMixin)

    # Infer decoder layers attribute name
    def _infer_decoder_layers_attr_name(model):
        model_class_name = model.__class__.__name__.lower()
        if "llama" in model_class_name:
            return "model.layers"
        elif "opt" in model_class_name:
            return "model.decoder.layers"
        elif "gpt" in model_class_name:
            return "transformer.h"
        else:
            return "model.layers"

    decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    # Wrap encoder with .visual attribute as expected by Flamingo
    flamingo_model = TimeSeriesFlamingoWithTrainableEncoder(
        vision_encoder=SimpleNamespace(visual=time_series_encoder),
        lang_encoder=lang_encoder,
        eoc_token_id=text_tokenizer.encode("<|endofchunk|>")[-1],
        media_token_id=text_tokenizer.encode("<image>")[-1],
        vis_dim=ENCODER_OUTPUT_DIM,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
    )

    # Create OpenTSLMFlamingo wrapper for proper data preprocessing
    opentslm_flamingo = OpenTSLMFlamingo(
        llm=flamingo_model,
        text_tokenizer=text_tokenizer,
        device=device
    )

    return opentslm_flamingo, text_tokenizer


def evaluate_model_on_mimiciv_mini100(
    model_id="OpenTSLM/llama-3.2-1b-ecg-flamingo",
    base_llm_id="meta-llama/Llama-3.2-1B",
    checkpoint_filename="softprompt-llama_3_2_1b-ecg.pt",
    dataset_path="data/ecg-qa-mini-100",
    ecg_data_path="data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0",
    results_base_dir="results_evaluation",
    device="cuda" if torch.cuda.is_available() else "cpu",
    max_patches=1024,
    cross_attn_every_n_layers=4,
):
    """
    Evaluate OpenTSLM Flamingo model on MIMIC-IV ECG-QA mini-100 dataset.
    """

    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = model_id.replace("/", "-")
    results_dir = Path(results_base_dir) / f"{model_name_safe}_mimiciv_mini100_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir)

    logger.info("=" * 60)
    logger.info("MIMIC-IV ECG-QA Mini-100 Evaluation")
    logger.info("=" * 60)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("")

    try:
        # ============================================================
        # 1. Load Model
        # ============================================================
        logger.info("=" * 60)
        logger.info(f"Loading model: {model_id}")
        logger.info("=" * 60)
        logger.info(f"Base LLM: {base_llm_id}")
        logger.info(f"Max patches: {max_patches} (matches checkpoint)")

        logger.info("Creating Flamingo model...")
        opentslm_flamingo, text_tokenizer = create_modified_opentslm_flamingo(
            device=device,
            llm_id=base_llm_id,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
            max_patches=max_patches,
        )

        logger.info("Downloading checkpoint from HuggingFace...")
        checkpoint_path = hf_hub_download(
            repo_id=model_id,
            filename=checkpoint_filename,
            cache_dir=os.path.expanduser("~/.cache/huggingface")
        )
        logger.info(f"Checkpoint downloaded to: {checkpoint_path}")

        logger.info("Loading checkpoint weights...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        opentslm_flamingo.llm.load_state_dict(checkpoint, strict=False)
        opentslm_flamingo.llm.eval()

        logger.info("✓ Model loaded successfully!")
        logger.info("")

        # ============================================================
        # 2. Load Dataset
        # ============================================================
        logger.info("=" * 60)
        logger.info("Loading dataset")
        logger.info("=" * 60)

        dataset = Mini100ECGQADataset(
            dataset_path=dataset_path,
            ecg_data_path=ecg_data_path,
            eos_token="<|end_of_text|>",
            logger=logger
        )

        logger.info(f"✓ Dataset loaded with {len(dataset)} samples")
        logger.info("")

        # ============================================================
        # 3. Evaluate
        # ============================================================
        logger.info("=" * 60)
        logger.info("Evaluating model")
        logger.info("=" * 60)

        all_results = []
        correct_count = 0
        question_type_stats = {}
        attribute_type_stats = {}

        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            sample = dataset[idx]

            try:
                # Generate prediction using OpenTSLMFlamingo's proper preprocessing
                batch = [sample]
                predictions = opentslm_flamingo.generate(batch, max_new_tokens=50)
                prediction = predictions[0] if predictions else ""

                # Get ground truth (remove EOS token for comparison)
                ground_truth = sample['answer'].replace("<|end_of_text|>", "").strip()
                prediction_clean = prediction.strip()

                # Check if correct (case-insensitive exact match)
                is_correct = prediction_clean.lower() == ground_truth.lower()
                if is_correct:
                    correct_count += 1

                # Track per-question-type accuracy
                question_type = sample.get('question_type', 'unknown')
                if question_type not in question_type_stats:
                    question_type_stats[question_type] = {'correct': 0, 'total': 0}
                question_type_stats[question_type]['total'] += 1
                if is_correct:
                    question_type_stats[question_type]['correct'] += 1

                # Track per-attribute-type accuracy
                attribute_type = sample.get('attribute_type', '')
                if attribute_type not in attribute_type_stats:
                    attribute_type_stats[attribute_type] = {'correct': 0, 'total': 0}
                attribute_type_stats[attribute_type]['total'] += 1
                if is_correct:
                    attribute_type_stats[attribute_type]['correct'] += 1

                # Store result
                result = {
                    'sample_id': idx,
                    'question': sample.get('question', ''),
                    'question_type': question_type,
                    'attribute_type': attribute_type,
                    'template_id': sample.get('template_id'),
                    'ground_truth': ground_truth,
                    'prediction': prediction_clean,
                    'is_correct': is_correct,
                    'ecg_id': sample.get('ecg_id'),
                }
                all_results.append(result)

            except Exception as e:
                logger.error(f"Error evaluating sample {idx}: {e}")
                import traceback
                logger.error(traceback.format_exc())

                result = {
                    'sample_id': idx,
                    'question': sample.get('question', ''),
                    'ground_truth': sample['answer'].replace("<|end_of_text|>", "").strip(),
                    'prediction': f"[ERROR: {str(e)}]",
                    'is_correct': False,
                }
                all_results.append(result)

        # ============================================================
        # 4. Save Results
        # ============================================================
        overall_accuracy = (correct_count / len(dataset)) * 100 if len(dataset) > 0 else 0

        logger.info("")
        logger.info("=" * 60)
        logger.info("Evaluation Complete")
        logger.info("=" * 60)
        logger.info(f"Total Samples: {len(dataset)}")
        logger.info(f"Correct: {correct_count}")
        logger.info(f"Overall Accuracy: {overall_accuracy:.2f}%")
        logger.info("")

        # Save detailed results
        with open(results_dir / "detailed_results.jsonl", 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')

        # Save metrics
        metrics = {
            'model': model_id,
            'base_llm': base_llm_id,
            'checkpoint': checkpoint_filename,
            'architecture': 'OpenTSLMFlamingo (TimeSeriesFlamingoWithTrainableEncoder)',
            'max_patches': max_patches,
            'dataset': 'MIMIC-IV ECG-QA mini-100',
            'dataset_path': dataset_path,
            'timestamp': timestamp,
            'device': device,
            'total_samples': len(dataset),
            'correct_predictions': correct_count,
            'overall_accuracy': overall_accuracy,
            'question_type_accuracy': {
                qt: {
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'accuracy': (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                }
                for qt, stats in question_type_stats.items()
            },
            'attribute_type_accuracy': {
                at: {
                    'correct': stats['correct'],
                    'total': stats['total'],
                    'accuracy': (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                }
                for at, stats in attribute_type_stats.items()
            }
        }

        with open(results_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save summary
        with open(results_dir / "summary.txt", 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("MIMIC-IV ECG-QA Mini-100 Evaluation Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Base LLM: {base_llm_id}\n")
            f.write(f"Architecture: OpenTSLMFlamingo (TimeSeriesFlamingoWithTrainableEncoder)\n")
            f.write(f"Checkpoint: {checkpoint_filename}\n")
            f.write(f"Max Patches: {max_patches}\n")
            f.write(f"Dataset: MIMIC-IV ECG-QA mini-100\n")
            f.write(f"Evaluation Date: {timestamp}\n")
            f.write(f"Device: {device}\n\n")
            f.write(f"Overall Results:\n")
            f.write(f"  Total Samples: {len(dataset)}\n")
            f.write(f"  Correct: {correct_count}\n")
            f.write(f"  Accuracy: {overall_accuracy:.2f}%\n\n")

            f.write("Per-Question-Type Accuracy:\n")
            for qt, stats in sorted(question_type_stats.items()):
                acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                f.write(f"  {qt}:\n")
                f.write(f"    Accuracy: {acc:.2f}%\n")
                f.write(f"    Correct/Total: {stats['correct']}/{stats['total']}\n\n")

            f.write("Per-Attribute-Type Accuracy:\n")
            for at, stats in sorted(attribute_type_stats.items()):
                acc = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
                f.write(f"  {at if at else '(empty)'}:\n")
                f.write(f"    Accuracy: {acc:.2f}%\n")
                f.write(f"    Correct/Total: {stats['correct']}/{stats['total']}\n\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("Example Predictions (first 10 samples)\n")
            f.write("=" * 60 + "\n\n")

            for i, result in enumerate(all_results[:10]):
                f.write(f"Sample {i+1}:\n")
                f.write(f"  Question: {result['question']}\n")
                f.write(f"  Ground Truth: {result['ground_truth']}\n")
                f.write(f"  Prediction: {result['prediction']}\n")
                f.write(f"  Correct: {'✓' if result['is_correct'] else '✗'}\n\n")

        logger.info(f"✓ Results saved to {results_dir}")
        logger.info("=" * 60)

        return metrics

    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    print("Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    metrics = evaluate_model_on_mimiciv_mini100()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2f}%")
    print(f"Correct: {metrics['correct_predictions']}/{metrics['total_samples']}")
    print("=" * 60)
