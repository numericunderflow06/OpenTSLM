#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import os
import subprocess
import json
import requests
import shutil
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from datasets import Dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from time_series_datasets.constants import RAW_DATA as RAW_DATA_PATH
from tqdm import tqdm


# MIMIC-IV-ECG Dataset
MIMICIV_ECG_BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0/"
MIMICIV_ECG_DIR = os.path.join(RAW_DATA_PATH, "mimic_iv_ecg")
MIMICIV_ECG_FILES_DIR = os.path.join(MIMICIV_ECG_DIR, "files")

# ECG-QA Repository (for MIMIC-IV questions)
ECG_QA_URL = "https://github.com/Jwoo5/ecg-qa"
ECG_QA_DIR = os.path.join(RAW_DATA_PATH, "ecg_qa")


def ensure_directory_exists(directory: str):
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def does_ecg_qa_exist():
    """Check if ECG-QA repository exists locally."""
    return os.path.exists(ECG_QA_DIR) and os.path.exists(os.path.join(ECG_QA_DIR, "ecgqa"))


def does_mimiciv_ecg_exist():
    """Check if MIMIC-IV-ECG dataset exists locally."""
    return (os.path.exists(MIMICIV_ECG_FILES_DIR) and
            os.path.exists(os.path.join(MIMICIV_ECG_DIR, "record_list.csv")) and
            os.path.exists(os.path.join(MIMICIV_ECG_DIR, "machine_measurements.csv")))


def clone_ecg_qa():
    """Clone the ECG-QA repository if not exists."""
    if not does_ecg_qa_exist():
        ensure_directory_exists(RAW_DATA_PATH)
        print(f"Cloning ECG-QA repository into {ECG_QA_DIR}...")
        subprocess.run([
            "git", "clone", ECG_QA_URL, ECG_QA_DIR
        ], check=True)
        print("ECG-QA repository cloned successfully!")


def download_ecg_qa_if_not_exists():
    """Download ECG-QA repository if it doesn't exist."""
    if not does_ecg_qa_exist():
        clone_ecg_qa()


def download_mimiciv_ecg():
    """Download MIMIC-IV-ECG dataset from PhysioNet.

    Note: This will download approximately 33.8 GB of data (90.4 GB uncompressed).
    The data will be stored in /local/home/wangni/OpenTSLM/data/mimic_iv_ecg/
    """
    ensure_directory_exists(RAW_DATA_PATH)
    ensure_directory_exists(MIMICIV_ECG_DIR)

    print(f"=" * 80)
    print(f"📥 MIMIC-IV-ECG Dataset Download")
    print(f"=" * 80)
    print(f"⚠️  WARNING: This will download approximately 33.8 GB of data")
    print(f"📂 Download location: {MIMICIV_ECG_DIR}")
    print(f"💾 Make sure you have at least 100 GB of free space in /local/home/wangni")
    print(f"=" * 80)

    # Check disk space
    total, used, free = shutil.disk_usage(RAW_DATA_PATH)
    free_gb = free / (1024**3)
    print(f"💿 Available disk space: {free_gb:.2f} GB")

    if free_gb < 100:
        print(f"❌ ERROR: Insufficient disk space. Need at least 100 GB, have {free_gb:.2f} GB")
        print(f"   Please free up space in /local/home/wangni before continuing.")
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB available, need 100 GB")

    print(f"✅ Sufficient disk space available")
    print()

    # Download using wget with recursive download
    print(f"🌐 Starting download from PhysioNet...")
    print(f"   URL: {MIMICIV_ECG_BASE_URL}")
    print(f"   This may take several hours depending on your connection speed.")
    print()

    try:
        # Use wget to recursively download the entire dataset
        # -r: recursive
        # -N: only download if newer (timestamp checking)
        # -c: continue partial downloads
        # -np: no parent (don't ascend to parent directory)
        # --no-check-certificate: skip certificate validation (PhysioNet sometimes has cert issues)
        # -P: prefix/directory to save to
        cmd = [
            "wget",
            "-r",  # recursive
            "-N",  # timestamping
            "-c",  # continue
            "-np",  # no parent
            "--no-check-certificate",  # skip cert check
            "--reject=index.html*",  # don't save directory listing pages
            "-P", MIMICIV_ECG_DIR,  # save to this directory
            "--cut-dirs=3",  # remove physionet.org/files/mimic-iv-ecg from path
            MIMICIV_ECG_BASE_URL
        ]

        print(f"🔧 Running command:")
        print(f"   {' '.join(cmd)}")
        print()

        subprocess.run(cmd, check=True)

        print()
        print(f"✅ Download completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed with error: {e}")
        print(f"   You can manually download from: {MIMICIV_ECG_BASE_URL}")
        print(f"   Or try using a different download method.")
        raise
    except FileNotFoundError:
        print(f"❌ wget not found. Please install wget or download manually.")
        print(f"   Manual download URL: {MIMICIV_ECG_BASE_URL}")
        print(f"   Alternative: Use your browser to download the ZIP file (33.8 GB)")
        raise

    # Verify that key files exist
    print()
    print(f"🔍 Verifying downloaded files...")

    required_files = [
        "record_list.csv",
        "machine_measurements.csv"
    ]

    for req_file in required_files:
        file_path = os.path.join(MIMICIV_ECG_DIR, req_file)
        if not os.path.exists(file_path):
            # Try looking in subdirectory created by wget
            alt_path = os.path.join(MIMICIV_ECG_DIR, "1.0", req_file)
            if os.path.exists(alt_path):
                print(f"   Found {req_file} in subdirectory, moving to root...")
                shutil.move(alt_path, file_path)
            else:
                raise FileNotFoundError(
                    f"Required MIMIC-IV-ECG file {req_file} not found in {MIMICIV_ECG_DIR}. "
                    f"Dataset may be incomplete or corrupted."
                )
        print(f"   ✓ {req_file}")

    # Check for files directory
    if not os.path.exists(MIMICIV_ECG_FILES_DIR):
        # Try looking in subdirectory created by wget
        alt_files_dir = os.path.join(MIMICIV_ECG_DIR, "1.0", "files")
        if os.path.exists(alt_files_dir):
            print(f"   Found files/ in subdirectory, moving to root...")
            shutil.move(alt_files_dir, MIMICIV_ECG_FILES_DIR)
        else:
            raise FileNotFoundError(
                f"MIMIC-IV-ECG files directory not found at {MIMICIV_ECG_FILES_DIR}. "
                f"Dataset may be incomplete or corrupted."
            )
    print(f"   ✓ files/ directory")

    print()
    print(f"✅ MIMIC-IV-ECG dataset verified successfully!")
    print(f"📂 Dataset location: {MIMICIV_ECG_DIR}")
    print(f"=" * 80)


def download_mimiciv_ecg_if_not_exists():
    """Download MIMIC-IV-ECG dataset if it doesn't exist."""
    if not does_mimiciv_ecg_exist():
        download_mimiciv_ecg()


def get_mimiciv_ecg_path(study_id: str, record_name: str) -> str:
    """Get the file path for a MIMIC-IV-ECG record.

    Args:
        study_id: Study ID (e.g., 's10000032')
        record_name: Record name (e.g., '02468199-5df1-7260-a90b-9e0ab56cb533')

    Returns:
        Base path to the ECG record (without .dat/.hea extension)
    """
    # MIMIC-IV-ECG structure: files/p{patient_id}/p{patient_id}-s{study_id}/{record_name}
    # Extract patient ID from study_id (e.g., s10000032 -> p10/p10000032)
    study_num = study_id[1:]  # Remove 's' prefix
    patient_id = f"p{study_num}"
    patient_dir = f"p{study_num[:2]}"  # First 2 digits for directory

    return os.path.join(
        MIMICIV_ECG_FILES_DIR,
        patient_dir,
        patient_id,
        study_id,
        record_name
    )


def load_mimiciv_metadata() -> pd.DataFrame:
    """Load MIMIC-IV-ECG metadata including machine measurements."""
    download_mimiciv_ecg_if_not_exists()

    record_list_path = os.path.join(MIMICIV_ECG_DIR, "record_list.csv")
    machine_measurements_path = os.path.join(MIMICIV_ECG_DIR, "machine_measurements.csv")

    if not os.path.exists(record_list_path):
        raise FileNotFoundError(f"MIMIC-IV-ECG record list not found: {record_list_path}")

    if not os.path.exists(machine_measurements_path):
        raise FileNotFoundError(f"MIMIC-IV-ECG machine measurements not found: {machine_measurements_path}")

    print("Loading MIMIC-IV-ECG metadata...")
    record_list = pd.read_csv(record_list_path)
    machine_measurements = pd.read_csv(machine_measurements_path)

    # Merge on study_id
    metadata_df = record_list.merge(machine_measurements, on='study_id', how='left')

    print(f"Loaded metadata for {len(metadata_df)} ECG records")
    return metadata_df


def create_mimiciv_clinical_context(study_id: str, metadata: pd.DataFrame) -> str:
    """Create safe clinical context from MIMIC-IV-ECG metadata.

    Args:
        study_id: Study ID (e.g., 's10000032')
        metadata: DataFrame with merged record_list and machine_measurements

    Returns:
        Clinical context string
    """
    # Find the ECG record
    record = metadata[metadata['study_id'] == study_id]

    if record.empty:
        return "12-lead ECG recording. Signal quality adequate for analysis."

    record = record.iloc[0]  # Get first match

    # Build context from safe metadata
    context_parts = []

    # ECG type
    context_parts.append("12-lead ECG")

    # ECG date/time (already de-identified)
    if pd.notna(record.get('ecg_time')):
        context_parts.append("clinical recording")

    # Machine measurements (technical features, not diagnostic)
    if pd.notna(record.get('heart_rate')):
        hr = int(record['heart_rate'])
        context_parts.append(f"measured heart rate: {hr} bpm")

    # Signal quality from machine measurements
    quality_parts = []
    if pd.notna(record.get('baseline_wander')):
        if str(record['baseline_wander']).strip().lower() in ['yes', 'true', '1']:
            quality_parts.append("baseline wander noted")

    if quality_parts:
        context_parts.append(f"Signal quality: {', '.join(quality_parts)}")
    else:
        context_parts.append("Signal quality: adequate for analysis")

    # Combine into natural sentence
    if len(context_parts) >= 2:
        context = f"{context_parts[0]}. " + ". ".join(context_parts[1:]) + "."
    else:
        context = ". ".join(context_parts) + "."

    return context


def load_ecg_qa_mimiciv_splits() -> Tuple[Dataset, Dataset, Dataset]:
    """Load ECG-QA MIMIC-IV splits as HuggingFace datasets.

    This loads the question-answer pairs from the ECG-QA repository and maps them
    to the corresponding MIMIC-IV-ECG signal files.
    """

    # Ensure both datasets exist
    download_ecg_qa_if_not_exists()
    download_mimiciv_ecg_if_not_exists()

    mimiciv_ecgqa_dir = os.path.join(ECG_QA_DIR, "ecgqa", "mimic-iv-ecg")

    if not os.path.exists(mimiciv_ecgqa_dir):
        raise FileNotFoundError(f"MIMIC-IV ECG-QA directory not found at {mimiciv_ecgqa_dir}")

    # Load the template version (paraphrased is also available)
    template_dir = os.path.join(mimiciv_ecgqa_dir, "template")

    def load_split_data(split_name: str) -> List[Dict]:
        """Load data for a specific split."""
        print(f"Loading {split_name} split...")
        split_dir = os.path.join(template_dir, split_name)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        all_data = []

        # Get all JSON files
        json_files = [f for f in os.listdir(split_dir) if f.endswith('.json')]

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in split directory: {split_dir}")

        # Load MIMIC-IV-ECG metadata once for this split
        mimiciv_metadata = load_mimiciv_metadata()

        # Load all JSON files in the split directory with progress bar
        for json_file in tqdm(json_files, desc=f"Loading {split_name} JSON files"):
            json_path = os.path.join(split_dir, json_file)

            try:
                with open(json_path, 'r') as f:
                    split_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in file {json_path}: {str(e)}")
            except Exception as e:
                raise RuntimeError(f"Failed to read JSON file {json_path}: {str(e)}")

            if not isinstance(split_data, list):
                raise ValueError(f"Expected JSON file {json_path} to contain a list, got {type(split_data)}")

            # Add ECG file paths and clinical context to each sample
            for sample_idx, sample in enumerate(tqdm(split_data, desc=f"Processing {json_file}", leave=False)):
                if not isinstance(sample, dict):
                    raise ValueError(f"Expected sample {sample_idx} in {json_path} to be a dict, got {type(sample)}")

                if "ecg_id" not in sample:
                    raise KeyError(f"Sample {sample_idx} in {json_path} missing required 'ecg_id' field")

                sample['ecg_paths'] = []
                sample['clinical_contexts'] = []
                ecg_ids = sample['ecg_id']

                if not isinstance(ecg_ids, list):
                    raise ValueError(f"Expected 'ecg_id' in sample {sample_idx} of {json_path} to be a list, got {type(ecg_ids)}")

                if not ecg_ids:
                    raise ValueError(f"Sample {sample_idx} in {json_path} has empty ecg_id list. "
                                   f"Every ECG-QA sample must have at least one ECG ID.")

                for ecg_id in ecg_ids:
                    if not isinstance(ecg_id, str):
                        raise ValueError(f"Expected ECG ID to be string, got {type(ecg_id)}: {ecg_id}")

                    # For MIMIC-IV, ecg_id format is "study_id/record_name"
                    # e.g., "s10000032/02468199-5df1-7260-a90b-9e0ab56cb533"
                    parts = ecg_id.split('/')
                    if len(parts) != 2:
                        raise ValueError(f"Invalid MIMIC-IV ECG ID format: {ecg_id}. Expected 'study_id/record_name'")

                    study_id, record_name = parts
                    ecg_path = get_mimiciv_ecg_path(study_id, record_name)

                    # Check if the ECG file exists
                    if os.path.exists(ecg_path + '.dat'):
                        sample['ecg_paths'].append(ecg_path + '.dat')
                    else:
                        # Missing ECG files - provide warning but don't fail
                        print(f"⚠️  Warning: ECG file not found: {ecg_path}.dat (ECG ID: {ecg_id})")
                        print(f"   Skipping this sample...")
                        continue

                    # Add safe clinical context
                    clinical_context = create_mimiciv_clinical_context(study_id, mimiciv_metadata)
                    sample['clinical_contexts'].append(clinical_context)

                # Only add sample if we found at least one valid ECG
                if sample['ecg_paths']:
                    all_data.append(sample)

        print(f"Loaded {len(all_data)} samples for {split_name} split")
        return all_data

    # Load each split
    train_data = load_split_data("train")
    val_data = load_split_data("valid")
    test_data = load_split_data("test")

    # Convert to HuggingFace datasets with progress
    print("Converting to HuggingFace datasets...")
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    print("Dataset loading complete!")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test the loader
    print("Testing MIMIC-IV-ECG loader...")

    # Test individual components
    print(f"ECG-QA exists: {does_ecg_qa_exist()}")
    print(f"MIMIC-IV-ECG exists: {does_mimiciv_ecg_exist()}")

    try:
        train, val, test = load_ecg_qa_mimiciv_splits()
        print(f"Loaded ECG-QA MIMIC-IV dataset:")
        print(f"  Train: {len(train)} samples")
        print(f"  Validation: {len(val)} samples")
        print(f"  Test: {len(test)} samples")

        if len(train) > 0:
            print(f"\nSample from training set:")
            sample = train[0]
            for key, value in sample.items():
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {key}: {value[:3]}... ({len(value)} items)")
                else:
                    print(f"  {key}: {value}")

            # Show clinical context
            if 'clinical_contexts' in sample and sample['clinical_contexts']:
                print(f"\nClinical context: {sample['clinical_contexts'][0]}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
