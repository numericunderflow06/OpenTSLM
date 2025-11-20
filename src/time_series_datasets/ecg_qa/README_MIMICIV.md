# MIMIC-IV-ECG Support for ECG-QA

This directory now includes support for the MIMIC-IV-ECG dataset, enabling cross-dataset evaluation between PTB-XL and MIMIC-IV.

## Overview

The implementation provides:
1. **MIMIC-IV-ECG signal loader** (`mimiciv_ecg_loader.py`) - Downloads and loads ECG waveforms from MIMIC-IV-ECG
2. **MIMIC-IV dataset class** (`ECGQAMimicIVDataset.py`) - Compatible with existing PTB-XL ECG-QA implementation

## Key Features

✅ **Fully compatible** with existing ECGQACoTQADataset (PTB-XL)
✅ **Same data format** - Both use WFDB format (.dat/.hea files)
✅ **Same preprocessing** - Downsample to 100Hz, normalize per lead
✅ **No credentials required** - MIMIC-IV-ECG v1.0 is open access
✅ **Safe downloads** - All data goes to `/local/home/wangni/OpenTSLM/data/`

## Dataset Information

### MIMIC-IV-ECG
- **Source:** PhysioNet (https://physionet.org/content/mimic-iv-ecg/1.0/)
- **Size:** 33.8 GB compressed, 90.4 GB uncompressed
- **Records:** ~800,000 ECGs from ~160,000 patients
- **Format:** WFDB (12-lead, 10 seconds, 500Hz)
- **License:** Open Data Commons Open Database License v1.0

### ECG-QA MIMIC-IV Questions
- **Source:** GitHub (https://github.com/Jwoo5/ecg-qa)
- **Templates:** 64 question templates (vs 70 for PTB-XL)
- **Excluded:** 6 templates (noise-related: 28-33, extra systoles: 35-38)
- **Splits:** Train/Valid/Test provided

## Directory Structure

After download, data will be organized as:

```
/local/home/wangni/OpenTSLM/data/
├── ecg_qa/                    # ECG-QA repository (questions)
│   └── ecgqa/
│       ├── ptbxl/            # PTB-XL questions
│       └── mimic-iv-ecg/     # MIMIC-IV questions ← NEW
│           ├── template/
│           │   ├── train/
│           │   ├── valid/
│           │   └── test/
│           ├── paraphrased/
│           ├── answers.csv
│           └── answers_for_each_template.csv
├── ptbxl/                     # PTB-XL ECG signals
└── mimic_iv_ecg/             # MIMIC-IV ECG signals ← NEW
    ├── files/                # ECG waveforms (.dat/.hea)
    ├── record_list.csv       # Record metadata
    └── machine_measurements.csv
```

## Usage

### Basic Usage

```python
from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset

# Load MIMIC-IV dataset (same API as PTB-XL)
train_dataset = ECGQAMimicIVDataset(
    split="train",
    EOS_TOKEN="</s>",
    max_samples=None,  # None for full dataset
    exclude_comparison=False,
    preload_processed_data=True,  # Cache for speed
    use_cot_format=True  # Compatible with PTB-XL CoT
)

# Access samples
sample = train_dataset[0]
print(sample.keys())
# -> dict_keys(['pre_prompt', 'time_series_text', 'post_prompt', 'answer', ...])
```

### Cross-Dataset Testing

```python
# Train on PTB-XL
from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
train_dataset = ECGQACoTQADataset(split="train", EOS_TOKEN="</s>")

# Test on MIMIC-IV
from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset
test_dataset = ECGQAMimicIVDataset(split="test", EOS_TOKEN="</s>")

# Both datasets have identical structure and format!
```

## Download Instructions

### Automatic Download (Recommended)

The dataset will be downloaded automatically when you first use it:

```python
from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset

# This will trigger automatic download if data doesn't exist
dataset = ECGQAMimicIVDataset(split="train", EOS_TOKEN="")
```

**Warning:** The automatic download is ~33.8 GB and may take several hours.

### Manual Download

```bash
cd /local/home/wangni/OpenTSLM

# Run safety check first
bash check_download_safety.sh

# Download MIMIC-IV-ECG manually
mkdir -p data/mimic_iv_ecg
cd data/mimic_iv_ecg
wget -r -N -c -np --no-check-certificate \
  --cut-dirs=3 \
  https://physionet.org/files/mimic-iv-ecg/1.0/

# Clone ECG-QA repo (includes MIMIC-IV questions)
cd /local/home/wangni/OpenTSLM/data
git clone https://github.com/Jwoo5/ecg-qa
```

### Testing Before Full Download

Test with a small subset first:

```python
# Test with just 5 samples (won't trigger full download)
from time_series_datasets.ecg_qa import mimiciv_ecg_loader

# Check if data exists
print("ECG-QA exists:", mimiciv_ecg_loader.does_ecg_qa_exist())
print("MIMIC-IV-ECG exists:", mimiciv_ecg_loader.does_mimiciv_ecg_exist())

# Clone ECG-QA repo only (small, ~few MB)
mimiciv_ecg_loader.download_ecg_qa_if_not_exists()
print("✅ ECG-QA questions downloaded")

# Don't download MIMIC-IV signals yet - wait until you're ready!
```

## Safety Notes

### ⚠️ CRITICAL: Download Location

**NEVER** let data download to `/home/wangni` - this will cause cluster storage quota issues and potential ban.

All data MUST go to `/local/home/wangni/OpenTSLM/data/`

**Verification before download:**
```bash
# Run safety check
bash /local/home/wangni/OpenTSLM/check_download_safety.sh

# Check available space (need 100+ GB)
df -h /local/home/wangni
```

### Disk Space Requirements

- **ECG-QA repo:** ~5 MB
- **MIMIC-IV-ECG download:** 33.8 GB (compressed)
- **MIMIC-IV-ECG extracted:** 90.4 GB
- **Recommended free space:** 100+ GB

### Download Time Estimates

- **ECG-QA questions:** < 1 minute
- **MIMIC-IV-ECG signals:** 2-6 hours (depending on network speed)

## Implementation Details

### Compatibility with PTB-XL

Both `ECGQACoTQADataset` (PTB-XL) and `ECGQAMimicIVDataset` (MIMIC-IV) share:

1. **Identical output format:**
   - `pre_prompt`: Task instructions with clinical context
   - `time_series_text`: List of TextTimeSeriesPrompt objects (12 leads)
   - `post_prompt`: Answer options and instructions
   - `answer`: Ground truth answer (or CoT rationale)

2. **Same preprocessing:**
   - Downsample from 500Hz to 100Hz
   - Normalize per lead (zero mean, unit variance)
   - 1000 samples per lead (10 seconds at 100Hz)

3. **Compatible caching:**
   - Both use class-level caches for efficiency
   - Caches can be shared for memory efficiency
   - `preload_processed_data=True` for maximum speed

### Differences from PTB-XL

| Feature | PTB-XL | MIMIC-IV |
|---------|--------|----------|
| **Question templates** | 70 | 64 |
| **Total ECGs** | 21,799 | ~800,000 |
| **Total questions** | 414,348 | TBD |
| **Labels** | Cardiologist-reviewed | Machine-generated |
| **Sampling rate** | 500Hz (some 100Hz) | 500Hz |
| **CoT data** | CSV files available | Not available (generated on-the-fly) |
| **Excluded templates** | None | 6 (noise & extra systoles) |

### Chain-of-Thought Format

MIMIC-IV dataset supports CoT format for compatibility:

```python
# Enable CoT format (default)
dataset = ECGQAMimicIVDataset(split="train", EOS_TOKEN="", use_cot_format=True)

# Disable CoT format (direct answers only)
dataset = ECGQAMimicIVDataset(split="train", EOS_TOKEN="", use_cot_format=False)
```

When `use_cot_format=True`, answers are formatted as:
```
"Based on the ECG analysis, the answer is: {answer}"
```

This ensures compatibility with models trained on PTB-XL CoT data.

## Testing

### Test the loader without downloading data:

```bash
cd /local/home/wangni/OpenTSLM
python src/time_series_datasets/ecg_qa/mimiciv_ecg_loader.py
```

### Test the dataset class:

```bash
cd /local/home/wangni/OpenTSLM
python src/time_series_datasets/ecg_qa/ECGQAMimicIVDataset.py
```

## Troubleshooting

### Issue: "wget: command not found"
```bash
# wget should be available on the cluster
which wget

# If not, try manual download from browser
# URL: https://physionet.org/files/mimic-iv-ecg/1.0/
```

### Issue: "Insufficient disk space"
```bash
# Check available space
df -h /local/home/wangni

# Clean up if needed
# (but NEVER touch /home/wangni - it's for small files only!)
```

### Issue: "Download stuck or slow"
```bash
# wget supports resume with -c flag
# Just re-run the command, it will continue from where it stopped
cd /local/home/wangni/OpenTSLM/data/mimic_iv_ecg
wget -r -N -c -np --no-check-certificate \
  --cut-dirs=3 \
  https://physionet.org/files/mimic-iv-ecg/1.0/
```

### Issue: "ECG file not found"
```bash
# Verify download completed
ls -lh /local/home/wangni/OpenTSLM/data/mimic_iv_ecg/files/

# Check record_list.csv exists
ls -lh /local/home/wangni/OpenTSLM/data/mimic_iv_ecg/record_list.csv

# If incomplete, re-run download with -c flag (continue)
```

## Citation

If you use the MIMIC-IV-ECG dataset, please cite:

```bibtex
@article{gow2020mimic,
  title={The MIMIC-IV-ECG database},
  author={Gow, Brian and others},
  journal={PhysioNet},
  year={2023}
}
```

If you use the ECG-QA dataset, please cite:

```bibtex
@article{oh2023ecgqa,
  title={ECG-QA: A Comprehensive Question Answering Dataset Combined With Electrocardiogram},
  author={Oh, Jungwoo and others},
  journal={NeurIPS},
  year={2023}
}
```

## Support

For issues related to:
- **MIMIC-IV-ECG data:** https://physionet.org/content/mimic-iv-ecg/1.0/
- **ECG-QA questions:** https://github.com/Jwoo5/ecg-qa
- **This implementation:** Create an issue in the OpenTSLM repository
