# Evaluation Comparison: PTB-XL vs MIMIC-IV
## Sanity Check Results for llama-3.2-1b-ecg-flamingo Checkpoint

**Date:** 2024-11-24  
**Purpose:** Compare checkpoint performance on original ECG-QA PTB-XL vs MIMIC-IV mini-100

---

## Summary

Both evaluations show **0% accuracy**, confirming that the poor performance on MIMIC-IV is consistent across datasets and likely indicates a bug in either:
1. The checkpoint itself
2. The model architecture/loading process
3. The evaluation setup

---

## Dataset Comparison

### PTB-XL Evaluation
- **Dataset:** ECG-QA CoT (Chain-of-Thought) - test split
- **Questions:** 100 (limited from 41,093 total)
- **Format:** Long CoT explanations ending with "Answer: ..."
- **Question Types:**
  - single-verify: 40 questions
  - single-choose: 25 questions
  - single-query: 35 questions
- **Results:** 0/100 correct (0.00% accuracy)
- **Evaluation Time:** ~1.4 minutes (~0.84 seconds per question)

### MIMIC-IV Evaluation  
- **Dataset:** MIMIC-IV ECG-QA mini-100
- **Questions:** 100
- **Format:** Short direct answers
- **Question Types:**
  - single-query: 23 questions
  - single-choose: 14 questions
  - comparison_consecutive-query: 23 questions
  - comparison_consecutive-verify: 19 questions
  - single-verify: 12 questions
  - comparison_irrelevant-verify: 7 questions
  - comparison_irrelevant-query: 2 questions
- **Results:** 0/100 correct (0.00% accuracy)

---

## Key Findings

### 1. Consistent 0% Accuracy
Both PTB-XL and MIMIC-IV evaluations achieved exactly 0% accuracy, indicating a systematic issue rather than dataset-specific problems.

### 2. Different Failure Modes

**PTB-XL (CoT format):**
- **Ground truth example:** 
  > "The ECG recording of this 82-year-old male patient shows... Therefore, based on the analysis of the ECG and the nature of the artifacts observed, the ECG does not show burst noise. Answer: no<|end_of_text|>"
- **Prediction example:** 
  > "The ECG recording of an 82-year-old male patient with a pacemaker shows evidence of baseline drift, static noise, and electrode artifacts, which are common issues that can affect the quality of the signal..."
- **Issue:** Predictions are cut off mid-sentence before reaching the answer (max_new_tokens=50 is too short for CoT responses)

**MIMIC-IV (short format):**
- **Ground truth example:** 
  > "atrial fibrillation<|end_of_text|>"
- **Prediction example:** 
  > ", supraventricular tachycardia, sinus tachycardia, supraventricular tachycardia, sinus tachycardia, sinus tachycardia, sinus tachycardia, sinus"
- **Issue:** Predictions are repetitive, nonsensical, and don't match the expected format

### 3. Model Generation Issues

The model generates text, but the outputs are:
- **Incoherent:** Repetitive patterns in MIMIC-IV
- **Incomplete:** Truncated before answers in PTB-XL
- **Off-target:** Not following the expected answer format

---

## Technical Details

### Model Configuration
- **Checkpoint:** `OpenTSLM/llama-3.2-1b-ecg-flamingo/softprompt-llama_3_2_1b-ecg.pt`
- **Base LLM:** meta-llama/Llama-3.2-1B
- **Architecture:** TimeSeriesFlamingoWithTrainableEncoder
- **Max Patches:** 1024
- **Checkpoint Epoch:** 12

### Evaluation Setup
- **Device:** CUDA
- **Generation Length:** max_new_tokens=50
- **Model loaded successfully:** ✓ (with 716 unexpected keys related to vision_encoder and perceiver)

---

## Possible Root Causes

1. **Checkpoint Mismatch:** The checkpoint may not be compatible with the evaluation architecture
2. **Training Issue:** The checkpoint may not have converged properly during training
3. **Generation Parameters:** max_new_tokens=50 is too short for CoT responses (should be 200-400 tokens)
4. **Tokenization Issue:** The special tokens or EOS handling may be incorrect
5. **Vision Encoder Issue:** The 716 unexpected keys suggest possible architecture mismatch

---

## Recommendations

1. **Increase max_new_tokens** to 200-400 for PTB-XL CoT evaluation
2. **Verify checkpoint training history** - check training logs/metrics from epoch 12
3. **Test with different checkpoints** from earlier/later epochs
4. **Check model generation** with simple prompts to isolate the issue
5. **Review architecture compatibility** between checkpoint and evaluation code
6. **Examine unexpected keys** in checkpoint loading - these may indicate missing/extra components

---

## Files Generated

### PTB-XL Evaluation
- **Results directory:** `results_evaluation/llama-3.2-1b-ecg-flamingo_ptbxl100_20251124_100457/`
- **Files:**
  - `detailed_results.jsonl` - Per-sample predictions and ground truth
  - `metrics.json` - Aggregated accuracy metrics
  - `summary.txt` - Human-readable summary
  - `evaluation.log` - Detailed execution log

### MIMIC-IV Evaluation
- **Results directory:** `results_evaluation/llama-3.2-1b-ecg-flamingo_mimiciv_mini100_20251123_194505/`
- **Files:** Same structure as PTB-XL

---

## Conclusion

The sanity check confirms that the 0% accuracy on MIMIC-IV is not dataset-specific. The checkpoint performs equally poorly on the original ECG-QA PTB-XL dataset, strongly suggesting an issue with the checkpoint itself, the model architecture, or the evaluation setup. The consistent failure across both datasets rules out data-specific bugs and points to a more fundamental problem that needs investigation.
