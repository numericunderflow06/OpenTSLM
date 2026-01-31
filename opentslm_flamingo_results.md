# OpenTSLM-Flamingo Results

## Model Accuracy

| Model | TSQA | HAR-CoT | Sleep-CoT | ECG-QA-CoT |
|-------|------|---------|-----------|------------|
| Llama3.2-1B (original) | 94.00 | 69.27 | 67.31 | 38.14 |
| Llama3.2-3B (original) | 90.10 | 69.03 | 69.14 | 46.25 |
| Llama3.2-1B (merged) | 94.81 | 74.93 | 70.22 | 45.52 |
| Llama3.2-3B (merged) | 92.52 | 72.60 | 71.34 | 49.41 |

## F1 Score

| Model | TSQA | HAR-CoT | Sleep-CoT | ECG-QA-CoT |
|-------|------|---------|-----------|------------|
| Llama3.2-1B (original) | 94.08 | 62.93 | 49.33 | 34.62 |
| Llama3.2-3B (original) | 90.14 | 62.77 | 45.45 | 40.25 |
| Llama3.2-1B (merged) | 94.05 | 65.52 | 57.32 | 37.14 |
| Llama3.2-3B (merged) | 92.73 | 69.31 | 52.13 | 41.55 |

## Test Set Sizes

| Dataset | Test Set Size |
|---------|---------------|
| TSQA | 4,800 |
| HAR-CoT | 8,222 |
| Sleep-CoT | 930 |
| ECG-QA-CoT | 41,093 |

### Complete Dataset Splits (Train / Validation / Test)

- **TSQA**: 38,400 / 4,800 / 4,800
- **HAR-CoT**: 68,542 / 8,718 / 8,222
- **Sleep-CoT**: 7,434 / 930 / 930
- **ECG-QA-CoT**: 159,313 / 31,137 / 41,093
