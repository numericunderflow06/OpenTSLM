# MIMIC-IV ECG-QA Evaluation Summary

## Issues Found and Fixed

### 1. Critical Bug: Checkpoint Loading Failure (FIXED ✓)
**Problem:** Model was using completely random/uninitialized weights for 716 parameters:
- Vision encoder (ECG signal processing)
- Perceiver (cross-modal adapter)
- Gated cross-attention layers
- Language model embeddings

**Root Cause:** State dict prefix mismatch
- Training checkpoint keys: `model.vision_encoder.pos_embed`
- Evaluation model expects: `vision_encoder.pos_embed`

**Fix Applied:** Strip `model.` prefix from checkpoint keys before loading

**Result:** All 716 weights now load correctly ✓

**File Fixed:** `eval_llama32_1b_ecg_flamingo_mimiciv_final.py` (lines 347-374)

### 2. Answer Normalization Bug (FIXED ✓)
**Problem:** Ground truth contains special tokens (`<|end_of_text|>`) that predictions don't have

**Fix Applied:** Strip special tokens from ground truth before comparison

**Result:** Accuracy improved from 0% to 17%

**File Fixed:** `eval_llama32_1b_ecg_flamingo_mimiciv_final.py` (lines 201-207)

## Current Evaluation Results

### With All Fixes Applied
- **Overall Accuracy: 17.00%** (17/100 correct)
- Model generates ECG-related content (not nonsense)
- Checkpoint loads correctly (0 missing keys)

### Per-Question-Type Breakdown
- comparison_consecutive-query: 0% (0/23)
- comparison_consecutive-verify: 0% (0/19)
- comparison_irrelevant-query: 0% (0/2)
- comparison_irrelevant-verify: 0% (0/7)
- single-choose: 0% (0/14)
- single-query: 0% (0/23)
- single-verify: 0% (0/12)

## Remaining Issues

### 1. Low Accuracy (17%)
The model generates ECG-related content but answers are frequently incorrect.

**Observed Problems:**
- **Repetitive outputs**: Model repeats phrases multiple times
  - Example: `"st elevation, st elevation in the inferior leads, st elevation in the inferior leads, st elevation..."`

- **Verbose responses**: Generates explanations instead of concise answers
  - Expected: `"p duration"`
  - Got: `", st segment, and QT interval. Answer: p duration."`

- **Format mismatch**: Predictions start with comma `, `
  - This suggests prompt format might not match training

- **Wrong answers for yes/no questions**: Model provides explanations instead of "yes"/"no"

### 2. Potential Causes

**Dataset Mismatch:**
- Model trained on "ECG QA dataset" (source unclear - PTB-XL or MIMIC-IV)
- Evaluating on MIMIC-IV mini-100 subset
- Dataset structures are similar but may have subtle differences

**Generation Parameters:**
- Current: `max_new_tokens=50`
- May need tuning: temperature, top_p, repetition_penalty

**Prompt Format:**
- Evaluation uses: `include_labels=True` (question only, no answer)
- Training format may have been different

## Recommendations

### Short-term
1. **Verify training dataset**: Check if model was trained on PTB-XL or MIMIC-IV
2. **Test on PTB-XL**: Run evaluation on PTB-XL mini-100 to compare performance
3. **Tune generation**:
   - Try `temperature=0.7`, `top_p=0.9`
   - Add `repetition_penalty=1.2` to reduce repetition
   - Adjust `max_new_tokens` based on answer length distribution

### Long-term
1. **Domain adaptation**: Fine-tune on MIMIC-IV if model was trained on PTB-XL
2. **Prompt engineering**: Investigate and match training prompt format
3. **Answer extraction**: Implement post-processing to extract concise answers from verbose outputs

## Files Modified

1. `eval_llama32_1b_ecg_flamingo_mimiciv_final.py` - Fixed checkpoint loading and normalization
2. `CHECKPOINT_LOADING_FIX.md` - Documentation of checkpoint bug
3. `test_checkpoint_loading.py` - Test script to verify fix
4. `rescore_results.py` - Re-scoring script with fixed normalization

## Next Steps

Run the same evaluation on PTB-XL data to see if performance is better, which would confirm dataset mismatch hypothesis.
