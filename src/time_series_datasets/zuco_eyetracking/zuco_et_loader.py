#
# ZuCo Eye-Tracking Dataset Loader for OpenTSLM
#
# Loads and preprocesses ZuCo 1.0 word-level eye-tracking data from MATLAB result files.
# Supports reading task classification (NR vs TSR) from gaze patterns.
#

import os
import sys
import pickle
from typing import Tuple, List, Dict

import numpy as np
from datasets import Dataset
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from time_series_datasets.constants import RAW_DATA

# --- Paths ---
ZUCO_BASE = "/home/wangni/tslm-co/data/zuco/raw/zuco1/q3zws/osfstorage"
CACHE_DIR = os.path.join(RAW_DATA, "zuco_et")
ET_CACHE_PKL = os.path.join(CACHE_DIR, "preprocessed_et.pkl")

# --- Subject configuration (same as EEG) ---
ALL_SUBJECTS = ["ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB", "ZKH", "ZKW", "ZMG", "ZPH"]
TRAIN_SUBJECTS = ["ZAB", "ZDM", "ZDN", "ZGW", "ZJM", "ZJN", "ZJS", "ZKB"]
VAL_SUBJECTS = ["ZKH", "ZKW"]
TEST_SUBJECTS = ["ZMG", "ZPH"]

# --- Eye-tracking metrics to extract ---
ET_METRICS = ["FFD", "GD", "GPT", "TRT", "nFixations"]
ET_METRIC_LABELS = {
    "FFD": "First Fixation Duration (ms) for each word",
    "GD": "Gaze Duration (ms) for each word",
    "GPT": "Go-Past Time (ms) for each word",
    "TRT": "Total Reading Time (ms) for each word",
    "nFixations": "Number of fixations for each word",
}

# --- Task directories ---
TASK_DIRS = {
    "NR": "task2 - NR/Matlab files",
    "TSR": "task3 - TSR/Matlab files",
}


def _load_mat_et(mat_path: str) -> List[Dict]:
    """
    Load word-level eye-tracking data from a ZuCo result .mat file.

    Returns list of dicts, one per sentence:
        {
            "sentence_text": str,
            "word_texts": list[str],
            "et_metrics": dict of metric_name -> list[float],
        }
    """
    from scipy.io import loadmat

    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    sentence_data = data.get("sentenceData", None)
    if sentence_data is None:
        print(f"  WARNING: No sentenceData in {mat_path}")
        return []

    if not hasattr(sentence_data, "__len__"):
        sentence_data = [sentence_data]

    sentences = []
    for sent in sentence_data:
        try:
            word_data = getattr(sent, "word", None)
            if word_data is None:
                continue

            if not hasattr(word_data, "__len__"):
                word_data = [word_data]

            word_texts = []
            metrics = {m: [] for m in ET_METRICS}

            for w in word_data:
                content = getattr(w, "content", "")
                if isinstance(content, str):
                    word_texts.append(content)
                else:
                    word_texts.append(str(content))

                for metric in ET_METRICS:
                    val = getattr(w, metric, None)
                    if val is None:
                        metrics[metric].append(0.0)
                    elif isinstance(val, np.ndarray):
                        if val.size == 0:
                            metrics[metric].append(0.0)
                        else:
                            v = float(val.flat[0])
                            metrics[metric].append(0.0 if np.isnan(v) else v)
                    else:
                        v = float(val)
                        metrics[metric].append(0.0 if np.isnan(v) else v)

            if len(word_texts) < 2:
                continue

            sentence_text = " ".join(word_texts)
            sentences.append({
                "sentence_text": sentence_text,
                "word_texts": word_texts,
                "et_metrics": metrics,
            })
        except Exception as e:
            continue

    return sentences


def preprocess_zuco_et(force: bool = False) -> Dict:
    """
    Process all subjects for NR and TSR tasks, extracting word-level ET data.
    Caches result to pickle.

    Returns dict:
        {
            task: {
                subject: [
                    {
                        "sentence_text": str,
                        "word_texts": list[str],
                        "et_metrics": dict of metric_name -> list[float],
                    },
                    ...
                ]
            }
        }
    """
    if os.path.exists(ET_CACHE_PKL) and not force:
        print(f"Loading cached ET data from {ET_CACHE_PKL}")
        with open(ET_CACHE_PKL, "rb") as f:
            return pickle.load(f)

    print("Preprocessing ZuCo Eye-Tracking data...")
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

            sentences = _load_mat_et(mat_path)
            all_data[task][subject] = sentences
            print(f"    {subject}: {len(sentences)} sentences")

    # Cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(ET_CACHE_PKL, "wb") as f:
        pickle.dump(all_data, f)
    print(f"\nSaved preprocessed ET to {ET_CACHE_PKL}")

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


def load_zuco_et_reading_task_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load ZuCo eye-tracking data for reading task classification (NR vs TSR).
    Subject-based split to prevent data leakage.

    Returns:
        Tuple of (train, validation, test) Dataset objects.
        Each sample has: et_channels (5 metric sequences), label, subject, sentence_text, word_texts
    """
    all_data = preprocess_zuco_et()

    samples = []
    for task, label in [("NR", "Normal Reading"), ("TSR", "Task-Specific Reading")]:
        if task not in all_data:
            print(f"WARNING: Task {task} not found in preprocessed ET data")
            continue
        for subject, sent_list in all_data[task].items():
            for sent in sent_list:
                # Convert ET metrics to list of channels (one per metric)
                et_channels = [sent["et_metrics"][m] for m in ET_METRICS]
                samples.append({
                    "et_channels": et_channels,
                    "label": label,
                    "subject": subject,
                    "sentence_text": sent["sentence_text"],
                    "word_texts": sent["word_texts"],
                    "task": task,
                })

    train_list, val_list, test_list = _split_by_subject(samples)

    print(f"\nET Reading Task splits:")
    print(f"  Train: {len(train_list)} (subjects: {TRAIN_SUBJECTS})")
    print(f"  Val:   {len(val_list)} (subjects: {VAL_SUBJECTS})")
    print(f"  Test:  {len(test_list)} (subjects: {TEST_SUBJECTS})")

    for split_name, split_data in [("Train", train_list), ("Val", val_list), ("Test", test_list)]:
        nr_count = sum(1 for s in split_data if s["label"] == "Normal Reading")
        tsr_count = sum(1 for s in split_data if s["label"] == "Task-Specific Reading")
        print(f"  {split_name}: NR={nr_count}, TSR={tsr_count}")

    return (
        Dataset.from_list(train_list),
        Dataset.from_list(val_list),
        Dataset.from_list(test_list),
    )


if __name__ == "__main__":
    print("=== ZuCo Eye-Tracking Data Preprocessing ===\n")

    train, val, test = load_zuco_et_reading_task_splits()
    print(f"\nTrain: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    if len(train) > 0:
        sample = train[0]
        print(f"\nSample keys: {list(sample.keys())}")
        print(f"ET channels: {len(sample['et_channels'])}")
        for i, m in enumerate(ET_METRICS):
            print(f"  {m}: length={len(sample['et_channels'][i])}")
        print(f"Label: {sample['label']}, Subject: {sample['subject']}")
        print(f"Sentence: {sample['sentence_text'][:80]}...")
        print(f"Words: {sample['word_texts'][:5]}...")
