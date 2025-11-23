# Dataset Structure Comparison: PTB-XL vs MIMIC-IV ECG-QA

## Overview

Both datasets use the same base class (`QADataset`) and follow identical formatting patterns for the Flamingo model. The key difference is that MIMIC-IV has an empty clinical context field.

---

## Raw Data Fields

### PTB-XL ECG-QA
```python
{
    'question': str,           # Question about the ECG
    'answer': str or list,     # Answer(s) to the question
    'clinical_contexts': list, # Clinical background information (populated)
    'ecg_paths': list,         # Paths to ECG .dat files
}
```

### MIMIC-IV ECG-QA
```python
{
    'question': str,           # Question about the ECG
    'answer': str or list,     # Answer(s) to the question
    'clinical_contexts': [""], # Empty - no clinical context available
    'ecg_paths': list,         # Paths to ECG .dat files
}
```

**Key Difference**: PTB-XL has actual clinical context data, MIMIC-IV uses empty strings.

---

## Formatted Sample Structure (After Dataset Processing)

Both datasets produce identical structure after processing:

```python
{
    'pre_prompt': str,          # Task description + clinical context + question
    'time_series': list,        # [12 ECG leads × 1000 samples each]
    'time_series_text': list,   # [12 labels, one per lead]
    'post_prompt': str,         # Instructions or answer options
    'answer': str,              # Ground truth answer with EOS token
}
```

---

## Prompt Structure Comparison

### PTB-XL

```
PRE-PROMPT:
You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context: [ACTUAL CLINICAL INFO HERE - patient demographics, symptoms, etc.]

Question: [Question about ECG characteristics]

Please analyze the ECG signal and provide your answer.

TIME SERIES INSERTION:
<image> This is ECG Lead I, it has mean X and std Y: <|endofchunk|>
<image> This is ECG Lead II, it has mean X and std Y: <|endofchunk|>
... (12 leads total)

POST-PROMPT:
Please provide your answer.
(or multiple choice options if applicable)

ANSWER:
[Ground truth answer]<|end_of_text|>
```

### MIMIC-IV

```
PRE-PROMPT:
You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context:
[EMPTY - no clinical context provided]

Question: [Question about ECG characteristics]

Please analyze the ECG signal and provide your answer.

TIME SERIES INSERTION:
<image> This is ECG Lead I, it has mean X and std Y: <|endofchunk|>
<image> This is ECG Lead II, it has mean X and std Y: <|endofchunk|>
... (12 leads total)

POST-PROMPT:
Please select your answer from the following options:
[Long list of multiple choice options for rhythm/diagnosis/etc.]

ANSWER:
[Ground truth answer]<|end_of_text|>
```

**Key Differences**:
1. PTB-XL has populated clinical context, MIMIC-IV shows "Clinical Context: " with nothing after it
2. MIMIC-IV typically has longer post-prompts with explicit multiple choice options
3. Both use identical time series token insertion pattern

---

## Time Series Data

### Both Datasets (Identical)

```python
time_series = [
    lead_I,    # numpy array, shape (1000,)
    lead_II,   # numpy array, shape (1000,)
    lead_III,  # numpy array, shape (1000,)
    lead_aVR,  # numpy array, shape (1000,)
    lead_aVL,  # numpy array, shape (1000,)
    lead_aVF,  # numpy array, shape (1000,)
    lead_V1,   # numpy array, shape (1000,)
    lead_V2,   # numpy array, shape (1000,)
    lead_V3,   # numpy array, shape (1000,)
    lead_V4,   # numpy array, shape (1000,)
    lead_V5,   # numpy array, shape (1000,)
    lead_V6,   # numpy array, shape (1000,)
]
```

- 12 ECG leads (standard 12-lead ECG)
- 1000 samples per lead
- Sampled at 100 Hz = 10 seconds of ECG data
- Data is normalized (mean/std calculated per lead)

---

## Time Series Labels

### Both Datasets (Identical Format)

```python
time_series_text = [
    "This is ECG Lead I, it has mean 0.0164 and std 0.0610:",
    "This is ECG Lead II, it has mean 0.0011 and std 0.0584:",
    "This is ECG Lead III, it has mean -0.0103 and std 0.0565:",
    "This is ECG Lead aVR, it has mean -0.0063 and std 0.0520:",
    "This is ECG Lead aVL, it has mean -0.0070 and std 0.0488:",
    "This is ECG Lead aVF, it has mean 0.0132 and std 0.0510:",
    "This is ECG Lead V1, it has mean -0.0168 and std 0.0668:",
    "This is ECG Lead V2, it has mean -0.0279 and std 0.1073:",
    "This is ECG Lead V3, it has mean -0.0422 and std 0.1768:",
    "This is ECG Lead V4, it has mean -0.0137 and std 0.1799:",
    "This is ECG Lead V5, it has mean 0.0060 and std 0.1737:",
    "This is ECG Lead V6, it has mean 0.0207 and std 0.1407:",
]
```

Each label provides:
- Lead name (I, II, III, aVR, aVL, aVF, V1-V6)
- Mean value (statistical normalization info)
- Standard deviation (statistical normalization info)

---

## Token Insertion Pattern

### Both Datasets (Identical)

The full prompt construction follows this pattern:

```
{pre_prompt} <image> {lead_1_label} <|endofchunk|> <image> {lead_2_label} <|endofchunk|> ... <image> {lead_12_label} <|endofchunk|> {post_prompt}
```

**Special Tokens**:
- `<image>` (token ID: 128257): Marks where model cross-attends to ECG signal
- `<|endofchunk|>` (token ID: 128256): Separates time series segments
- `<|end_of_text|>` (token ID: 128001): EOS token for answer

**Insertion Count**: 12 `<image>` tokens (one per ECG lead)

---

## Example Samples

### PTB-XL Example

**Question**: "What is the heart rate shown in this ECG?"

**Pre-prompt**:
```
You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context: 68-year-old female patient presenting with chest pain and shortness of breath.

Question: What is the heart rate shown in this ECG?

Please analyze the ECG signal and provide your answer.
```

**Post-prompt**:
```
Please provide your answer.
```

**Answer**: `75 bpm<|end_of_text|>`

---

### MIMIC-IV Example

**Question**: "What are the rhythm-related indications that can be seen on this ECG?"

**Pre-prompt**:
```
You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context:

Question: What are the rhythm-related indications that can be seen on this ECG?

Please analyze the ECG signal and provide your answer.
```

**Post-prompt**:
```
Please select your answer from the following options:
AV sequential pacemaker, AV-junctional rhythm, accelerated idioventricular rhythm, accelerated junctional rhythm, atrial arrhythmia, atrial bigeminy, atrial couplet, atrial fibrillation, atrial flutter, atrial tachycardia, dual chamber electronic pacing, ectopic atrial bradycardia, ectopic atrial rhythm, ectopic atrial tachycardia, electronic atrial pacing, extreme tachycardia, fusion complexes, idioventricular rhythm, junctional bradycardia, junctional rhythm, junctional tachycardia, non-sustained ventricular tachycardia, none, pacemaker activity, pacemaker rhythm, paroxysmal idioventricular rhythm, rapid ventricular response, regular rhythm, sinus arrhythmia, sinus bradycardia, sinus rhythm, sinus tachycardia, slow ventricular response, supraventricular bigeminy BIGU bigeminal pattern, supraventricular rhythm, supraventricular tachycardia, uncontrolled ventricular response, ventricular bigeminy, ventricular couplet, ventricular escape rhythm, ventricular tachycardia, ventricular trigeminy, ventricular-paced complexes or rhythm, wandering pacemaker, wide QRS tachycardia
```

**Answer**: `atrial fibrillation<|end_of_text|>`

---

## Token Count Comparison

### PTB-XL (Estimated)
- Pre-prompt: ~80-120 tokens (includes clinical context)
- Post-prompt: ~10-50 tokens (usually brief)
- Time series labels: ~240 tokens (12 × ~20)
- Special tokens: 24 (12 × 2)
- **Total: ~350-430 tokens**

### MIMIC-IV (Measured)
- Pre-prompt: 49 tokens (no clinical context)
- Post-prompt: 290 tokens (long multiple choice list)
- Time series labels: ~240 tokens (12 × ~20)
- Special tokens: 24 (12 × 2)
- **Total: ~603 tokens**

**Key Difference**: MIMIC-IV has longer post-prompts (multiple choice options), PTB-XL has longer pre-prompts (clinical context).

---

## Implementation Notes

### Dataset Classes

Both inherit from `QADataset` base class:

```python
# PTB-XL
class ECGQADataset(QADataset):
    def __init__(self, split, EOS_TOKEN, ...):
        super().__init__(...)
        # Loads PTB-XL specific data

# MIMIC-IV
class ECGQAMimicIVDataset(QADataset):
    def __init__(self, split, EOS_TOKEN, ...):
        super().__init__(...)
        # Loads MIMIC-IV specific data
```

### Data Loading

**PTB-XL**: Uses `load_ecg_qa_ptbxl_splits()` from `ptbxl_loader.py`

**MIMIC-IV**: Uses `load_ecg_qa_mimiciv_splits()` from `mimiciv_ecg_loader.py`

### ECG File Format

Both use WFDB format:
- `.dat` file: Binary ECG signal data
- `.hea` file: Header with metadata

---

## Summary

| Feature | PTB-XL | MIMIC-IV |
|---------|---------|----------|
| Clinical Context | ✅ Populated | ❌ Empty |
| Multiple Choice Options | Sometimes | Usually |
| Prompt Length | Shorter (~350-430 tokens) | Longer (~603 tokens) |
| Time Series Format | Identical | Identical |
| Token Insertion | Identical | Identical |
| Base Class | QADataset | QADataset |
| ECG Leads | 12 leads × 1000 samples | 12 leads × 1000 samples |

**Bottom Line**: The datasets use identical formatting and token insertion patterns. The only substantive difference is PTB-XL includes clinical context while MIMIC-IV does not.
