# Evaluation Pipeline Issues and Fixes

## Critical Issues Identified

### 1. Model Structure Mismatch (HIGH PRIORITY) ⚠️

**Location**: `eval_llama32_1b_ecg_flamingo_mimiciv_final.py:280-291, 348-360`

**Problem**:
- Evaluation creates raw `TimeSeriesFlamingoWithTrainableEncoder` directly
- Checkpoint was likely saved from `OpenTSLMFlamingo` wrapper
- State dict key mismatch causes parameters to be silently skipped with `strict=False`

**Current Code**:
```python
flamingo_model = TimeSeriesFlamingoWithTrainableEncoder(
    vision_encoder=SimpleNamespace(visual=time_series_encoder),
    lang_encoder=lang_encoder,
    ...
)
# ...
missing_keys, unexpected_keys = flamingo_model.load_state_dict(ckpt['model_state'], strict=False)
```

**Issue**: Missing/unexpected keys logged but not investigated. Critical cross-attention weights may not be loaded.

**Root Cause (CONFIRMED)**:
- Checkpoint keys have `model.` and `llm.` prefixes
- Example: `model.vision_encoder.pos_embed`, `model.perceiver.latents`, `llm.lang_encoder.layers.0.weight`
- Model expects: `vision_encoder.pos_embed`, `perceiver.latents`, `lang_encoder.layers.0.weight`
- This prefix mismatch causes ALL keys to be "missing" with `strict=False`

**Fix**:
- Strip `model.` prefix when loading checkpoint keys
- Remap `llm.` prefix to `lang_encoder.` prefix
- Verify no missing keys for critical components (perceiver, gated_cross_attn_layers)
- Add assertion to fail if critical keys are missing

---

### 2. Hacky Wrapper Object (MEDIUM PRIORITY) ⚠️

**Location**: `eval_llama32_1b_ecg_flamingo_mimiciv_final.py:427-436`

**Problem**:
```python
temp_wrapper = type('TempWrapper', (), {
    'text_tokenizer': self.text_tokenizer,
    'device': self.device,
    'llm': self.llm
})()

input_ids, images, attention_mask, _ = OpenTSLMFlamingo.pad_and_apply_batch(
    temp_wrapper, batch, include_labels=True
)
```

Creates fake wrapper object to access `pad_and_apply_batch`. Fragile and error-prone.

**Fix**:
- Use `OpenTSLMFlamingo` instance directly
- Call instance method instead of hacking a temporary object

---

### 3. Missing Generation Parameters (HIGH PRIORITY) ⚠️

**Location**: `eval_llama32_1b_ecg_flamingo_mimiciv_final.py:440-447`

**Problem**:
```python
gen_ids = self.llm.generate(
    vision_x=images,
    lang_x=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,  # Only 50!
    eos_token_id=self.text_tokenizer.eos_token_id,
    pad_token_id=self.text_tokenizer.pad_token_id,
    # Missing: do_sample, temperature, repetition_penalty
)
```

**Issues**:
- No `do_sample` parameter (defaults to greedy decoding)
- No `temperature` for sampling diversity
- No `repetition_penalty` to prevent loops
- `max_new_tokens=50` may be too short
- Model might immediately generate EOS with greedy decoding

**Fix**:
- Add generation parameters:
  - `do_sample=True` (or test both greedy and sampling)
  - `temperature=0.7` or `1.0`
  - `max_new_tokens=100`
  - `repetition_penalty=1.0` or `1.1`

---

### 4. Tokenizer Encoding Inconsistency (LOW PRIORITY) ⚠️

**Location**: `eval:284` vs `OpenTSLMFlamingo.py:191-194`

**Problem**:
```python
# Eval script uses encode():
media_token_id = text_tokenizer.encode("<image>")[-1]

# OpenTSLMFlamingo uses __call__():
media_token_id = tokenizer("<image>", add_special_tokens=False)["input_ids"][-1]
```

**Issue**: Different tokenization methods might produce different token IDs.

**Fix**: Use consistent tokenization method matching OpenTSLMFlamingo.

---

### 5. No Checkpoint Verification (MEDIUM PRIORITY) ⚠️

**Location**: `eval:352-357`

**Problem**:
```python
missing_keys, unexpected_keys = flamingo_model.load_state_dict(ckpt['model_state'], strict=False)
if missing_keys:
    logger.warning(f"Missing keys: {missing_keys}")  # Only warns!
```

**Issue**: Missing keys are logged but execution continues. Critical weights might not be loaded.

**Fix**:
- Check for critical missing keys: `perceiver`, `gated_cross_attn_layers`, `vision_encoder`
- Fail if critical components are missing
- Log detailed statistics about what was/wasn't loaded

---

### 6. Cache Directory Not Enforced (MEDIUM PRIORITY) ⚠️

**Location**: `eval:341-345`

**Problem**:
```python
checkpoint_path = hf_hub_download(
    repo_id="OpenTSLM/llama-3.2-1b-ecg-flamingo",
    filename="softprompt-llama_3_2_1b-ecg.pt",
    cache_dir="/local/home/wangni/.cache/huggingface"  # Specified here
)
```

But earlier in `create_modified_opentslm_flamingo`:
```python
text_tokenizer = AutoTokenizer.from_pretrained(
    llm_id,
    local_files_only=False,
    trust_remote_code=True,
    # NO cache_dir specified! Will use default ~/.cache
)
```

**Issue**: Tokenizer and base model download to `/home/wangni/.cache` instead of `/local/home/wangni/.cache`.

**Fix**: Add `cache_dir="/local/home/wangni/.cache/huggingface"` to all `from_pretrained` calls.

---

## Recommended Implementation Plan

1. ✅ **Create new evaluation script** using `OpenTSLMFlamingo` directly
2. ✅ **Add cache_dir** to all model/tokenizer loading calls
3. ✅ **Use proper checkpoint loading** from OpenTSLMFlamingo's `load_from_file` method
4. ✅ **Add generation parameters** (temperature, do_sample, etc.)
5. ✅ **Add debug logging** for first sample to inspect tokenization and generation
6. ✅ **Verify checkpoint loading** with assertions for critical keys

---

## Expected Behavior After Fixes

### Before:
- Predictions: Empty strings or whitespace
- Accuracy: 0%
- No errors reported

### After Fixes:
- Predictions: Actual medical diagnoses or "none"
- Accuracy: >0% (even if low for zero-shot)
- Clear error messages if checkpoint loading fails
- Debug logs showing actual token IDs and generated text

---

## Testing Strategy

1. **Run with debug logging** on first 5 samples
2. **Check checkpoint loading** - verify no missing critical keys
3. **Inspect tokenization** - print input_ids for first sample
4. **Test generation parameters**:
   - Greedy (do_sample=False)
   - Sampling (do_sample=True, temperature=0.7)
   - Higher temperature (temperature=1.0)
5. **Compare outputs** between different generation strategies

---

## Notes

- The checkpoint is from: https://huggingface.co/OpenTSLM/llama-3.2-1b-ecg-flamingo
- File: `softprompt-llama_3_2_1b-ecg.pt`
- Should be cached in: `/local/home/wangni/.cache/huggingface`
- Base model: `meta-llama/Llama-3.2-1B`
- Dataset: MIMIC-IV ECG-QA mini-100 (100 questions)

---

## References

- Evaluation script: `eval_llama32_1b_ecg_flamingo_mimiciv_final.py`
- Model wrapper: `src/model/llm/OpenTSLMFlamingo.py`
- Flamingo core: `src/model/llm/TimeSeriesFlamingoWithTrainableEncoder.py`
- Dataset: `src/time_series_datasets/ecg_qa/ECGQAMimicIVDataset.py`
