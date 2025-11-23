# Zero-Shot Evaluation Results: OpenTSLM Llama-3.2-1B ECG Flamingo on MIMIC-IV ECG-QA

## Overview

This document summarizes the zero-shot evaluation of the `OpenTSLM/llama-3.2-1b-ecg-flamingo` model on the MIMIC-IV ECG-QA mini-100 dataset.

## Model Information

- **Model**: OpenTSLM/llama-3.2-1b-ecg-flamingo
- **HuggingFace**: https://huggingface.co/OpenTSLM/llama-3.2-1b-ecg-flamingo
- **Base LLM**: meta-llama/Llama-3.2-1B
- **Architecture**: OpenTSLMFlamingo (TimeSeriesFlamingoWithTrainableEncoder)
- **Checkpoint**: softprompt-llama_3_2_1b-ecg.pt
- **Max Patches**: 1024 (matches checkpoint configuration)

## Dataset Information

- **Dataset**: MIMIC-IV ECG-QA mini-100
- **Source**: `data/ecg-qa-mini-100/paraphrased/train/000000.json`
- **Total Samples**: 100 questions
- **ECG Format**: 12-lead ECG, 1000 samples per lead (10 seconds at 100 Hz)
- **Question Types**:
  - comparison_consecutive-query: 23 samples
  - comparison_consecutive-verify: 19 samples
  - comparison_irrelevant-query: 2 samples
  - comparison_irrelevant-verify: 7 samples
  - single-choose: 14 samples
  - single-query: 23 samples
  - single-verify: 12 samples

## Evaluation Setup

- **Evaluation Date**: 2025-11-23 15:55:22
- **Device**: CUDA (GPU)
- **Batch Size**: 1
- **Patch Size**: 128
- **Max New Tokens**: 100
- **Generation Mode**: Zero-shot (no fine-tuning on MIMIC-IV)
- **Evaluation Script**: `eval_llama32_1b_ecg_flamingo_mimiciv_final.py`

## Results

### Overall Performance

```
Overall Accuracy: 0.00% (0/100)
Correct Predictions: 0
Total Samples: 100
```

### Per-Question-Type Accuracy

| Question Type | Correct/Total | Accuracy |
|--------------|---------------|----------|
| comparison_consecutive-query | 0/23 | 0.00% |
| comparison_consecutive-verify | 0/19 | 0.00% |
| comparison_irrelevant-query | 0/2 | 0.00% |
| comparison_irrelevant-verify | 0/7 | 0.00% |
| single-choose | 0/14 | 0.00% |
| single-query | 0/23 | 0.00% |
| single-verify | 0/12 | 0.00% |

## Issues Identified

### 1. Empty Predictions

The primary issue is that the model generates **empty or near-empty predictions** for most samples:

- ~90% of predictions are completely empty (`""`)
- ~10% contain garbage text or repeated prompt fragments

### 2. Example Predictions

**Sample 1** (single-query):
- **Question**: "What are the rhythm-related indications that can be seen on this ECG?"
- **Ground Truth**: "atrial fibrillation"
- **Prediction**: `""` (empty)

**Sample 2** (single-choose):
- **Question**: "What is the diagnostic symptom indicated by this ECG, specifically Myocardial infarction in inferior leads or AV-junctional rhythm, including any uncertain symptoms?"
- **Ground Truth**: "Myocardial infarction in inferior leads"
- **Prediction**: `" "` (single space)

**Sample 4** (comparison - with garbage output):
- **Question**: "Is ST Elevation still accounted for in the recent tracing, in comparison to the previous one?"
- **Ground Truth**: "yes"
- **Prediction**: `" ɵ ɵ This is ECG Lead I (Recording 3), it has mean 0.0171 and std 0.1399: ɵ ɵ This is ECG Lead II (Recording 3), it has mean -"`
- **Note**: Model is repeating ECG lead labels instead of answering

**Sample 6** (single-verify - with garbage output):
- **Question**: "Is lead V3 showing indications of Deep S wave in this ECG?"
- **Ground Truth**: "no"
- **Prediction**: `" ısıt ısıt This is ECG Lead V7, it has mean 0.0212 and std 0.1886: ısıt ısıt This is ECG Lead V8, it has mean 0.0081 and std"`
- **Note**: Repeating lead labels with garbled tokens

## Potential Causes

### 1. Model-Dataset Mismatch

The model was trained on ECG-QA with PTB-XL data, which includes clinical context:
- **PTB-XL**: Prompts include clinical context (patient demographics, symptoms)
- **MIMIC-IV**: Prompts have empty clinical context field

This distribution shift may confuse the model.

### 2. Prompt Format Differences

While both datasets use identical token insertion patterns, MIMIC-IV has:
- Empty clinical context sections
- Longer post-prompts with extensive multiple-choice options
- Different question phrasings

### 3. Generation Parameters

The model may need different generation settings:
- Current: `max_new_tokens=100`, basic greedy decoding
- May need: temperature adjustment, different sampling strategies, or prompt engineering

### 4. Checkpoint Compatibility

Possible issues:
- Checkpoint may expect different input format
- Checkpoint may have been trained with different special token handling
- Cross-attention mechanism may not be properly initialized for inference

## Data Pipeline Validation

The data pipeline has been **thoroughly validated** and is working correctly:

✅ **ECG Data Loading**:
- All 151 unique ECG files successfully loaded
- 12 leads × 1000 samples per ECG
- Proper normalization applied

✅ **Prompt Formatting**:
- Pre-prompt, post-prompt, and time series labels correctly formatted
- 12 `<image>` tokens inserted (one per ECG lead)
- Special tokens (`<|endofchunk|>`, `<|end_of_text|>`) properly added

✅ **Token Insertion**:
- Verified to match PTB-XL ECG-QA format exactly
- Same base class (`QADataset`) used for both datasets

See `DATASET_STRUCTURE_COMPARISON.md` for detailed comparison.

## Evaluation Files

### Scripts
- `eval_llama32_1b_ecg_flamingo_mimiciv_final.py` - Main evaluation script
- `show_prompt_examples.py` - Demonstrates prompt structure with token insertion
- `debug_dataset_structure.py` - Verifies dataset formatting

### Documentation
- `DATASET_STRUCTURE_COMPARISON.md` - PTB-XL vs MIMIC-IV comparison
- `EVALUATION_RESULTS.md` - This file

### Results
- `results_evaluation/llama-3.2-1b-ecg-flamingo_mimiciv_mini100_20251123_155522/`
  - `detailed_results.jsonl` - Per-sample predictions and ground truth
  - `metrics.json` - Structured metrics
  - `summary.txt` - Human-readable summary

### Logs
- `eval_run_WORKING.log` - Complete evaluation log (151KB)

## Recommendations

### For Reproducibility

1. **Verify Model Checkpoint**: Confirm checkpoint is compatible with inference code
2. **Check Training Details**: Review how the model was trained (prompt format, generation settings)
3. **Compare with PTB-XL**: Run same evaluation on PTB-XL ECG-QA to establish baseline

### For Improvement

1. **Prompt Engineering**:
   - Add clinical context (even if synthetic)
   - Adjust prompt format to match training distribution
   - Test different instruction phrasings

2. **Generation Settings**:
   - Experiment with temperature (0.1, 0.7, 1.0)
   - Try different sampling strategies (top-p, top-k)
   - Adjust max_new_tokens

3. **Fine-tuning**:
   - Fine-tune on MIMIC-IV ECG-QA training set
   - Use few-shot prompting with examples

## Technical Details

### Hardware
- GPU: CUDA-enabled device
- Evaluation time: ~65 seconds for 100 samples (~1.5 samples/sec)

### Software Versions
- PyTorch: [version from environment]
- Transformers: [version from environment]
- Python: 3.x

### Model Loading Configuration

```python
opentslm_flamingo = OpenTSLMFlamingo(
    llm=flamingo_model,  # TimeSeriesFlamingoWithTrainableEncoder
    text_tokenizer=text_tokenizer,
    device='cuda'
)
```

**Key Parameters**:
- `max_patches=1024` (matches checkpoint)
- `lang_encoder_name='meta-llama/Llama-3.2-1B'`
- `cross_attn_every_n_layers=4`
- `only_attend_immediate_media=False`

## Conclusion

The evaluation pipeline is **functioning correctly**, but the model is **not generating meaningful predictions** on MIMIC-IV ECG-QA. The 0% accuracy is due to:

1. Empty/whitespace predictions (~90% of samples)
2. Garbage output repeating prompt fragments (~10% of samples)

This is likely a **model-dataset compatibility issue** rather than a data pipeline problem. The model appears to be trained for a different distribution (PTB-XL with clinical context) and struggles with MIMIC-IV's format.

**Next Steps**:
1. Evaluate on PTB-XL ECG-QA for baseline
2. Investigate checkpoint training details
3. Experiment with prompt engineering and generation settings
4. Consider fine-tuning on MIMIC-IV if zero-shot is not viable

## References

- Model: https://huggingface.co/OpenTSLM/llama-3.2-1b-ecg-flamingo
- ECG-QA Dataset: https://github.com/Jwoo5/ecg-qa
- MIMIC-IV ECG: https://physionet.org/content/mimic-iv-ecg/
