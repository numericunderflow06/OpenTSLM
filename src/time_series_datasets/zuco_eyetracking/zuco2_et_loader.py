#
# ZuCo 2.0 Eye-Tracking Dataset Loader for OpenTSLM
#
# Loads and preprocesses ZuCo 2.0 word-level eye-tracking data from MATLAB v7.3 result files.
# Uses h5py (HDF5) instead of scipy.io.loadmat because ZuCo 2.0 .mat files are MATLAB v7.3 format.
# Supports reading task classification (NR vs TSR) from gaze patterns.
#

import os
import sys
import pickle
from typing import Tuple, List, Dict

import numpy as np
import h5py
from datasets import Dataset
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from time_series_datasets.constants import RAW_DATA

# --- Paths ---
ZUCO2_BASE = "/home/wangni/tslm-co/data/zuco/raw/zuco2/2urht/osfstorage"
CACHE_DIR = os.path.join(RAW_DATA, "zuco2_et")
ET_CACHE_PKL = os.path.join(CACHE_DIR, "preprocessed_et_zuco2.pkl")

# --- Subject configuration (18 total, excluding YMH/YRH/YMS from 16 benchmark) ---
ALL_SUBJECTS = [
    "YAC", "YAG", "YAK", "YDG", "YDR", "YFR", "YFS", "YHS",
    "YIS", "YLS", "YMD", "YRK", "YRP", "YSD", "YSL", "YTL",
]
TRAIN_SUBJECTS = ["YAC", "YAG", "YAK", "YDG", "YDR", "YFR", "YFS", "YHS", "YIS", "YLS"]
VAL_SUBJECTS = ["YMD", "YRK", "YRP"]
TEST_SUBJECTS = ["YSD", "YSL", "YTL"]

# --- Eye-tracking metrics to extract (same as ZuCo 1.0) ---
ET_METRICS = ["FFD", "GD", "GPT", "TRT", "nFixations"]
ET_METRIC_LABELS = {
    "FFD": "First Fixation Duration (ms) for each word",
    "GD": "Gaze Duration (ms) for each word",
    "GPT": "Go-Past Time (ms) for each word",
    "TRT": "Total Reading Time (ms) for each word",
    "nFixations": "Number of fixations for each word",
}

# --- Task directories (ZuCo 2.0 uses task1=NR, task2=TSR) ---
TASK_DIRS = {
    "NR": "task1 - NR/Matlab files",
    "TSR": "task2 - TSR/Matlab files",
}


def _read_hdf5_string(f: h5py.File, ref) -> str:
    """Dereference an HDF5 object reference and decode a uint16 char array to string."""
    obj = f[ref]
    data = obj[:]
    if obj.dtype == np.uint16:
        return "".join(chr(c) for c in data.flatten())
    return str(data)


def _read_hdf5_scalar(f: h5py.File, ref) -> float:
    """
    Dereference an HDF5 object reference and read a scalar value.
    ZuCo 2.0 uses uint64 dtype to indicate missing fixation data -> return 0.0.
    """
    obj = f[ref]
    if obj.dtype.kind == "u":  # unsigned int (uint64) means no fixation
        return 0.0
    val = obj[()]
    if isinstance(val, np.ndarray):
        if val.size == 0:
            return 0.0
        v = float(val.flat[0])
        return 0.0 if np.isnan(v) else v
    v = float(val)
    return 0.0 if np.isnan(v) else v


def _load_mat_et(mat_path: str) -> List[Dict]:
    """
    Load word-level eye-tracking data from a ZuCo 2.0 result .mat file (HDF5 format).

    Returns list of dicts, one per sentence:
        {
            "sentence_text": str,
            "word_texts": list[str],
            "et_metrics": dict of metric_name -> list[float],
        }
    """
    try:
        f = h5py.File(mat_path, "r")
    except Exception as e:
        print(f"  WARNING: Cannot open {mat_path}: {e}")
        return []

    try:
        if "sentenceData" not in f:
            print(f"  WARNING: No sentenceData in {mat_path}")
            return []

        sd = f["sentenceData"]
        n_sentences = sd["content"].shape[0]

        sentences = []
        for si in range(n_sentences):
            try:
                # Read sentence text
                content_ref = sd["content"][si, 0]
                sentence_text_raw = _read_hdf5_string(f, content_ref)

                # Read word data
                word_ref = sd["word"][si, 0]
                word_data = f[word_ref]

                if "content" not in word_data:
                    continue

                n_words = word_data["content"].shape[0]
                if n_words < 2:
                    continue

                word_texts = []
                metrics = {m: [] for m in ET_METRICS}

                for wi in range(n_words):
                    # Read word text
                    wc_ref = word_data["content"][wi, 0]
                    word_text = _read_hdf5_string(f, wc_ref)
                    word_texts.append(word_text)

                    # Read each metric
                    for metric in ET_METRICS:
                        if metric in word_data:
                            m_ref = word_data[metric][wi, 0]
                            val = _read_hdf5_scalar(f, m_ref)
                            metrics[metric].append(val)
                        else:
                            metrics[metric].append(0.0)

                sentence_text = " ".join(word_texts)
                sentences.append({
                    "sentence_text": sentence_text,
                    "word_texts": word_texts,
                    "et_metrics": metrics,
                })
            except Exception:
                continue

        return sentences
    finally:
        f.close()


def preprocess_zuco2_et(force: bool = False) -> Dict:
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
        print(f"Loading cached ZuCo 2.0 ET data from {ET_CACHE_PKL}")
        with open(ET_CACHE_PKL, "rb") as f:
            return pickle.load(f)

    print("Preprocessing ZuCo 2.0 Eye-Tracking data...")
    all_data = {}

    for task, task_dir in TASK_DIRS.items():
        print(f"\nProcessing task: {task}")
        all_data[task] = {}

        task_path = os.path.join(ZUCO2_BASE, task_dir)
        if not os.path.exists(task_path):
            print(f"  WARNING: Task directory not found: {task_path}")
            print(f"  Run the download script to fetch {task} Matlab files from OSF.")
            continue

        for subject in tqdm(ALL_SUBJECTS, desc=f"  {task} subjects"):
            mat_file = f"results{subject}_{task}.mat"
            mat_path = os.path.join(task_path, mat_file)

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
    print(f"\nSaved preprocessed ZuCo 2.0 ET to {ET_CACHE_PKL}")

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


def load_zuco2_et_reading_task_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load ZuCo 2.0 eye-tracking data for reading task classification (NR vs TSR).
    Subject-based split to prevent data leakage.

    Returns:
        Tuple of (train, validation, test) Dataset objects.
        Each sample has: et_channels (5 metric sequences), label, subject, sentence_text, word_texts, task
    """
    all_data = preprocess_zuco2_et()

    samples = []
    for task, label in [("NR", "Normal Reading"), ("TSR", "Task-Specific Reading")]:
        if task not in all_data:
            print(f"WARNING: Task {task} not found in preprocessed ZuCo 2.0 ET data")
            continue
        for subject, sent_list in all_data[task].items():
            for sent in sent_list:
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

    print(f"\nZuCo 2.0 ET Reading Task splits:")
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
    print("=== ZuCo 2.0 Eye-Tracking Data Preprocessing ===\n")

    train, val, test = load_zuco2_et_reading_task_splits()
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
