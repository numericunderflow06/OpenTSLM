# Understanding MIMIC-IV-ECG Data for OpenTSLM

## Overview
This document explains the MIMIC-IV-ECG dataset structure, download process, and how it's used in the OpenTSLM project for ECG question-answering tasks.

## Dataset Information

**MIMIC-IV-ECG Dataset:**
- Source: PhysioNet (https://physionet.org/files/mimic-iv-ecg/1.0/)
- Size: ~33.8 GB compressed, ~90.4 GB uncompressed
- Content: 12-lead ECG recordings from MIMIC-IV database
- Format: WFDB (WaveForm DataBase) format with .dat and .hea files

**ECG-QA Dataset:**
- Source: https://github.com/Jwoo5/ecg-qa
- Content: Question-answer pairs mapped to ECG recordings
- Splits: train/valid/test in JSON format

## Data Download

### Download Script Location
`src/time_series_datasets/ecg_qa/mimiciv_ecg_loader.py`

### Download Process
The download uses `wget` with recursive crawling:
```bash
wget -r -N -c -np --no-check-certificate --reject=index.html* --cut-dirs=3 \
  https://physionet.org/files/mimic-iv-ecg/1.0/
```

**Key flags:**
- `-r`: Recursive download
- `-N`: Timestamping (skip already downloaded files)
- `-c`: Continue partial downloads
- `-np`: No parent directory
- `--reject=index.html*`: **Important** - Prevents downloading HTML directory listings
- `--cut-dirs=3`: Removes `physionet.org/files/mimic-iv-ecg` from local path

### Index.html Issue (Resolved)
**Problem:** During recursive wget download, 8,589+ index.html files were being created (directory listings from the web server).

**Solution:**
1. Added `--reject=index.html*` flag to wget command (line 121 in mimiciv_ecg_loader.py)
2. Deleted all existing index.html files
3. Added to .gitignore: `**/index.html`, `data/`, `mimic_download.log`, `nohup.out`

**Result:** wget now downloads index.html as .tmp files and immediately removes them.

## Directory Structure

### Downloaded Data Structure
```
data/mimic_iv_ecg/physionet.org/files/
├── p1000/                          # Patient directory (first 2 digits: p10)
│   ├── RECORDS                     # List of all recordings in this directory
│   ├── p10000032/                  # Individual patient subdirectory
│   │   ├── s40689238/              # Study ID directory
│   │   │   ├── 40689238.dat        # Binary ECG signal data
│   │   │   └── 40689238.hea        # Header file (metadata)
│   │   ├── s44458630/
│   │   └── ...
│   └── ...
├── p1001/
├── ...
└── p1999/
```

### RECORDS Files
**Purpose:** Standard PhysioNet format for cataloging recordings in a directory.

**Format:** Plain text, one path per line (without file extensions):
```
p10000032/s40689238/40689238
p10000032/s44458630/44458630
p10000117/s45090959/45090959
```

**Usage:**
- Quick enumeration without directory traversal
- Standard PhysioNet/WFDB tooling compatibility
- Data integrity verification
- Note: OpenTSLM loader doesn't use these directly (uses ECG-QA JSON mappings instead)

## File Formats

### ECG Signal Files
- **`.dat` files:** Binary ECG signal data in WFDB format (multi-lead time series)
- **`.hea` files:** ASCII header with metadata:
  - Sampling rate
  - Number of leads
  - Duration
  - Signal amplitude units
  - Recording date/time (de-identified)

### Metadata Files
- `record_list.csv` - Complete list of all ECG records with basic info
- `machine_measurements.csv` - Machine-computed measurements (heart rate, intervals, etc.)

## Data Processing Pipeline

### Raw Data → Ready-to-Use Dataset

**1. Download Phase** (automatic via `download_mimiciv_ecg_if_not_exists()`):
- Downloads all .dat, .hea, RECORDS, and CSV files
- Stores in `data/mimic_iv_ecg/`

**2. ECG-QA Integration** (automatic via `download_ecg_qa_if_not_exists()`):
- Clones ECG-QA repository
- Provides JSON files with question-answer pairs
- Maps questions to specific ECG IDs

**3. Loading Phase** (via `load_ecg_qa_mimiciv_splits()`):
- Reads ECG-QA JSON files from `data/ecg_qa/ecgqa/mimic-iv-ecg/template/{train,valid,test}/`
- For each sample:
  - Extracts ECG ID (format: `study_id/record_name`)
  - Maps to actual file path: `files/p{XX}/p{patient_id}/s{study_id}/{record_name}.dat`
  - Loads metadata from CSVs
  - Creates safe clinical context (de-identified)
  - Checks if file exists, skips if missing
- Returns HuggingFace Dataset objects with:
  - `question`: Question text
  - `answer`: Answer text
  - `ecg_id`: List of ECG identifiers
  - `ecg_paths`: List of paths to .dat files
  - `clinical_contexts`: List of clinical context strings

**Key Code:** `src/time_series_datasets/ecg_qa/mimiciv_ecg_loader.py:298-414`

### No Manual Processing Required
The loader handles all processing automatically. Just call:
```python
from time_series_datasets.ecg_qa.mimiciv_ecg_loader import load_ecg_qa_mimiciv_splits

train_dataset, val_dataset, test_dataset = load_ecg_qa_mimiciv_splits()
```

## Download Status Monitoring

### Check Download Progress
```bash
# View live download log
tail -f /local/home/wangni/OpenTSLM/mimic_download.log

# Check wget process status
ps aux | grep wget | grep mimic

# Count downloaded ECG files
find data/mimic_iv_ecg -name "*.dat" | wc -l

# Check total download size
du -sh data/mimic_iv_ecg/
```

### Expected Download Time
Given the 33.8 GB dataset size and recursive crawling through 1000s of nested directories, expect **several hours** for complete download.

**Download phases:**
1. Directory structure crawling (~1 hour) - Creates dirs, downloads RECORDS files
2. Bulk file download (~several hours) - Downloads actual .dat/.hea ECG signal files

## Important Paths

### Data Locations
- MIMIC-IV-ECG data: `/local/home/wangni/OpenTSLM/data/mimic_iv_ecg/`
- ECG-QA repository: `/local/home/wangni/OpenTSLM/data/ecg_qa/`
- Download log: `/local/home/wangni/OpenTSLM/mimic_download.log`

### Code Locations
- Main loader: `src/time_series_datasets/ecg_qa/mimiciv_ecg_loader.py`
- ECG-QA loader: `src/time_series_datasets/ecg_qa/ecgqa_loader.py`
- Constants: `src/time_series_datasets/constants.py` (defines `RAW_DATA` path)

## gitignore Configuration

To prevent committing large data files:
```gitignore
# Data directories
raw_data
data/

# Download artifacts
**/index.html
mimic_download.log
nohup.out
```

## Troubleshooting

### If Download Stops
The download process uses `-c` (continue) flag, so you can safely restart:
```bash
cd /local/home/wangni/OpenTSLM/data/mimic_iv_ecg
wget -r -N -c -np --no-check-certificate --reject=index.html* --cut-dirs=3 \
  https://physionet.org/files/mimic-iv-ecg/1.0/
```

### If index.html Files Reappear
Delete them:
```bash
find /local/home/wangni/OpenTSLM/data/mimic_iv_ecg -name "index.html" -type f -delete
```

### Check Data Integrity
After download completes:
```python
from src.time_series_datasets.ecg_qa.mimiciv_ecg_loader import (
    does_mimiciv_ecg_exist,
    does_ecg_qa_exist
)

print(f"MIMIC-IV-ECG exists: {does_mimiciv_ecg_exist()}")
print(f"ECG-QA exists: {does_ecg_qa_exist()}")
```

## References

- PhysioNet MIMIC-IV-ECG: https://physionet.org/content/mimic-iv-ecg/1.0/
- ECG-QA Paper: https://arxiv.org/abs/2105.04040
- ECG-QA GitHub: https://github.com/Jwoo5/ecg-qa
- WFDB Format: https://physionet.org/physiotools/wfdb.shtml

---
*Document created: 2025-11-20*
*Last updated: 2025-11-20*
