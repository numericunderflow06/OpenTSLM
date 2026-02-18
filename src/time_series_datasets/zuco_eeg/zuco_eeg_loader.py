#
# ZuCo EEG Dataset Loader for OpenTSLM
#
# Loads and preprocesses ZuCo 1.0 EEG data from MATLAB result files.
# Supports two tasks:
#   1. Reading task classification (NR vs TSR)
#   2. Sentiment classification from EEG during reading (SR task)
#

import os
import sys
import csv
import pickle
from typing import Tuple, List, Dict, Optional

import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from time_series_datasets.constants import RAW_DATA

# --- Paths ---
ZUCO_BASE = "/home/wangni/tslm-co/data/zuco/raw/zuco1/q3zws/osfstorage"
CACHE_DIR = os.path.join(RAW_DATA, "zuco_eeg")
EEG_CACHE_PKL = os.path.join(CACHE_DIR, "preprocessed_eeg.pkl")

SENTIMENT_CSV = os.path.join(ZUCO_BASE, "task_materials", "sentiment_labels_task1.csv")

# --- Subject configuration ---
ALL_SUBJECTS = ["ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH"]
TRAIN_SUBJECTS = ["ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB"]
VAL_SUBJECTS = ["ZKH", "ZKW"]
TEST_SUBJECTS = ["ZMG", "ZPH"]

# --- EEG processing config ---
SELECTED_CHANNELS = [0, 13, 26, 39, 52, 65, 78, 91]  # 8 evenly spaced from 105
DOWNSAMPLE_FACTOR = 5  # 500Hz -> 100Hz

# --- Task directories ---
TASK_DIRS = {
    "SR": "task1- SR/Matlab files",
    "NR": "task2 - NR/Matlab files",
    "TSR": "task3 - TSR/Matlab files",
}


def _load_mat_eeg(mat_path: str) -> List[Dict]:
    """
    Load EEG data from a ZuCo result .mat file.

    Returns list of dicts, one per sentence:
        {
            "sentence_text": str,
            "raw_eeg": np.ndarray (105, timepoints) or None,
            "word_texts": list[str],
        }
    """
    from scipy.io import loadmat

    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # The results file has a sentenceData array
    sentence_data = data.get("sentenceData", None)
    if sentence_data is None:
        print(f"  WARNING: No sentenceData in {mat_path}")
        return []

    # Handle case where sentenceData is a single struct (not array)
    if not hasattr(sentence_data, "__len__"):
        sentence_data = [sentence_data]

    sentences = []
    for sent in sentence_data:
        try:
            # Get raw EEG data
            raw_eeg = getattr(sent, "rawData", None)
            if raw_eeg is None or not hasattr(raw_eeg, "shape"):
                continue

            # rawData should be (105, timepoints) or similar
            if raw_eeg.ndim != 2:
                continue

            # Ensure shape is (channels, timepoints) using known channel count
            n_channels_expected = 105
            if raw_eeg.shape[0] == n_channels_expected:
                pass  # Already (channels, timepoints)
            elif raw_eeg.shape[1] == n_channels_expected:
                raw_eeg = raw_eeg.T  # Was (timepoints, channels)
            else:
                continue  # Neither dimension matches expected channel count

            if raw_eeg.shape[0] < max(SELECTED_CHANNELS) + 1:
                continue

            # Get sentence text from word content
            word_data = getattr(sent, "word", None)
            word_texts = []
            if word_data is not None:
                if not hasattr(word_data, "__len__"):
                    word_data = [word_data]
                for w in word_data:
                    content = getattr(w, "content", "")
                    if isinstance(content, str):
                        word_texts.append(content)

            sentence_text = " ".join(word_texts)

            sentences.append({
                "sentence_text": sentence_text,
                "raw_eeg": raw_eeg,
                "word_texts": word_texts,
            })
        except Exception as e:
            continue

    return sentences


def _select_channels(eeg: np.ndarray, channel_indices: List[int]) -> np.ndarray:
    """Select specific channels from full EEG array. Returns (n_channels, timepoints)."""
    return eeg[channel_indices, :]


def _downsample(eeg: np.ndarray, factor: int) -> np.ndarray:
    """Downsample EEG by taking every factor-th sample. Input/output: (n_channels, timepoints)."""
    return eeg[:, ::factor]


def _normalize_text_key(text: str) -> str:
    """Normalize sentence text for matching: lowercase, strip whitespace."""
    return text.lower().strip()


def _load_sentiment_labels() -> Dict[str, Dict]:
    """
    Load sentiment labels from CSV, keyed by normalized sentence text.

    Uses text-based matching instead of index-based, because the .mat files
    may contain sentences not present in the CSV (causing index shifts).
    """
    labels = {}
    with open(SENTIMENT_CSV, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) < 4:
                continue
            if row[0].startswith("#"):
                continue
            sentence_text = row[1]
            control = row[2].strip()
            sentiment_label = int(row[3])
            text_key = _normalize_text_key(sentence_text)
            labels[text_key] = {
                "sentence_text": sentence_text,
                "sentiment_label": sentiment_label,
                "control": control,
            }
    return labels


def preprocess_zuco_eeg(force: bool = False) -> Dict:
    """
    Process all subjects for all tasks, extracting and preprocessing EEG.
    Caches result to pickle.

    Returns dict:
        {
            task: {
                subject: [
                    {
                        "sentence_text": str,
                        "eeg_channels": np.ndarray (8, variable_timepoints),
                        "sentence_index": int,
                    },
                    ...
                ]
            }
        }
    """
    if os.path.exists(EEG_CACHE_PKL) and not force:
        print(f"Loading cached EEG data from {EEG_CACHE_PKL}")
        with open(EEG_CACHE_PKL, "rb") as f:
            return pickle.load(f)

    print("Preprocessing ZuCo EEG data (this may take a while)...")
    all_data = {}

    for task, task_dir in TASK_DIRS.items():
        print(f"\nProcessing task: {task}")
        all_data[task] = {}

        for subject in tqdm(ALL_SUBJECTS, desc=f"  {task} subjects"):
            mat_file = f"results{subject}_{task}.mat"
            mat_path = os.path.join(ZUCO_BASE, task_dir, mat_file)

            if not os.path.exists(mat_path):
                print(f"  WARNING: File not found: {mat_path}")
                continue

            sentences = _load_mat_eeg(mat_path)
            processed = []

            for i, sent in enumerate(sentences):
                raw_eeg = sent["raw_eeg"]
                # Select channels
                selected = _select_channels(raw_eeg, SELECTED_CHANNELS)
                # Downsample
                downsampled = _downsample(selected, DOWNSAMPLE_FACTOR)

                # Skip very short signals (< 4 timepoints after downsampling, PATCH_SIZE=4)
                if downsampled.shape[1] < 4:
                    continue

                # Check for NaN/Inf
                if np.any(np.isnan(downsampled)) or np.any(np.isinf(downsampled)):
                    # Replace NaN/Inf with 0
                    downsampled = np.nan_to_num(downsampled, nan=0.0, posinf=0.0, neginf=0.0)

                processed.append({
                    "sentence_text": sent["sentence_text"],
                    "eeg_channels": downsampled,
                    "sentence_index": i,
                })

            all_data[task][subject] = processed
            print(f"    {subject}: {len(processed)} sentences")

    # Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(EEG_CACHE_PKL, "wb") as f:
        pickle.dump(all_data, f)
    print(f"\nSaved preprocessed EEG to {EEG_CACHE_PKL}")

    return all_data


def _split_by_subject(
    samples: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples by subject into train/val/test."""
    train, val, test = [], [], []
    for sample in samples:
        subj = sample["subject"]
        if subj in TRAIN_SUBJECTS:
            train.append(sample)
        elif subj in VAL_SUBJECTS:
            val.append(sample)
        elif subj in TEST_SUBJECTS:
            test.append(sample)
    return train, val, test


def load_zuco_eeg_reading_task_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load ZuCo EEG data for reading task classification (NR vs TSR).
    Subject-based split to prevent data leakage.

    Returns:
        Tuple of (train, validation, test) Dataset objects.
        Each sample has: eeg_channels (list of lists), label, subject, sentence_text
    """
    all_data = preprocess_zuco_eeg()

    samples = []
    for task, label in [("NR", "Normal Reading"), ("TSR", "Task-Specific Reading")]:
        if task not in all_data:
            print(f"WARNING: Task {task} not found in preprocessed data")
            continue
        for subject, sent_list in all_data[task].items():
            for sent in sent_list:
                # Convert numpy arrays to lists for Dataset serialization
                eeg_channels = [ch.tolist() for ch in sent["eeg_channels"]]
                samples.append({
                    "eeg_channels": eeg_channels,
                    "label": label,
                    "subject": subject,
                    "sentence_text": sent["sentence_text"],
                    "task": task,
                })

    train_list, val_list, test_list = _split_by_subject(samples)

    print(f"\nEEG Reading Task splits:")
    print(f"  Train: {len(train_list)} (subjects: {TRAIN_SUBJECTS})")
    print(f"  Val:   {len(val_list)} (subjects: {VAL_SUBJECTS})")
    print(f"  Test:  {len(test_list)} (subjects: {TEST_SUBJECTS})")

    # Count labels
    for split_name, split_data in [("Train", train_list), ("Val", val_list), ("Test", test_list)]:
        nr_count = sum(1 for s in split_data if s["label"] == "Normal Reading")
        tsr_count = sum(1 for s in split_data if s["label"] == "Task-Specific Reading")
        print(f"  {split_name}: NR={nr_count}, TSR={tsr_count}")

    return (
        Dataset.from_list(train_list),
        Dataset.from_list(val_list),
        Dataset.from_list(test_list),
    )


def load_zuco_eeg_sentiment_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load ZuCo EEG data for sentiment classification (SR task only).
    Subject-based split to prevent data leakage.
    Labels: negative (-1), neutral (0), positive (1)

    Returns:
        Tuple of (train, validation, test) Dataset objects.
        Each sample has: eeg_channels, label, subject, sentence_text, sentiment_value
    """
    all_data = preprocess_zuco_eeg()
    sentiment_labels = _load_sentiment_labels()

    SENTIMENT_MAP = {-1: "negative", 0: "neutral", 1: "positive"}

    if "SR" not in all_data:
        raise RuntimeError("SR task not found in preprocessed EEG data")

    samples = []
    skipped_no_label = 0
    skipped_control = 0

    for subject, sent_list in all_data["SR"].items():
        for sent in sent_list:
            eeg_text_key = _normalize_text_key(sent["sentence_text"])

            # Primary: exact text match
            label_info = sentiment_labels.get(eeg_text_key)

            # Fallback: prefix match (handles encoding artifacts in .mat word data)
            if label_info is None:
                for csv_key, csv_info in sentiment_labels.items():
                    if eeg_text_key[:20] == csv_key[:20]:
                        label_info = csv_info
                        break

            if label_info is None:
                skipped_no_label += 1
                continue

            # Skip control sentences
            if label_info["control"].upper() == "CONTROL":
                skipped_control += 1
                continue

            sentiment_value = label_info["sentiment_label"]
            label_str = SENTIMENT_MAP[sentiment_value]

            eeg_channels = [ch.tolist() for ch in sent["eeg_channels"]]
            samples.append({
                "eeg_channels": eeg_channels,
                "label": label_str,
                "subject": subject,
                "sentence_text": sent["sentence_text"],
                "sentiment_value": sentiment_value,
            })

    print(f"\nEEG Sentiment: skipped {skipped_no_label} (no label), {skipped_control} (control)")

    train_list, val_list, test_list = _split_by_subject(samples)

    print(f"\nEEG Sentiment splits:")
    print(f"  Train: {len(train_list)} (subjects: {TRAIN_SUBJECTS})")
    print(f"  Val:   {len(val_list)} (subjects: {VAL_SUBJECTS})")
    print(f"  Test:  {len(test_list)} (subjects: {TEST_SUBJECTS})")

    for split_name, split_data in [("Train", train_list), ("Val", val_list), ("Test", test_list)]:
        for label in ["negative", "neutral", "positive"]:
            count = sum(1 for s in split_data if s["label"] == label)
            print(f"  {split_name} {label}: {count}")

    return (
        Dataset.from_list(train_list),
        Dataset.from_list(val_list),
        Dataset.from_list(test_list),
    )


if __name__ == "__main__":
    print("=== ZuCo EEG Data Preprocessing ===\n")

    print("--- Reading Task (NR vs TSR) ---")
    train, val, test = load_zuco_eeg_reading_task_splits()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    if len(train) > 0:
        sample = train[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"EEG channels: {len(sample['eeg_channels'])}, first channel length: {len(sample['eeg_channels'][0])}")
        print(f"Label: {sample['label']}, Subject: {sample['subject']}")
        print(f"Sentence: {sample['sentence_text'][:80]}...")

    print("\n--- Sentiment Classification ---")
    train_s, val_s, test_s = load_zuco_eeg_sentiment_splits()
    print(f"Train: {len(train_s)}, Val: {len(val_s)}, Test: {len(test_s)}")
    if len(train_s) > 0:
        sample = train_s[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Label: {sample['label']}, Sentiment value: {sample['sentiment_value']}")
        print(f"Sentence: {sample['sentence_text'][:80]}...")
