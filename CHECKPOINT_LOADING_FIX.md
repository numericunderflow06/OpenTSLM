# Checkpoint Loading Bug Fix

## Problem Summary

The evaluation script was generating nonsense outputs because **almost all model weights were missing** during checkpoint loading. The model was using random/uninitialized weights for:
- All vision encoder parameters
- All perceiver parameters
- All gated cross-attention layers
- Embedding tokens and language model head

## Root Cause

**State Dict Prefix Mismatch:**

During training, the model is wrapped in `OpenTSLMFlamingo` class which stores the Flamingo model as `self.model`:
```python
self.model = TimeSeriesFlamingoWithTrainableEncoder(...)
```

When the checkpoint is saved, all weights get a `model.` prefix:
- Checkpoint: `model.vision_encoder.pos_embed`
- Checkpoint: `model.perceiver.latents`
- Checkpoint: `model.lang_encoder.model.layers.0.gated_cross_attn_layer.*`

The evaluation script creates `TimeSeriesFlamingoWithTrainableEncoder` directly (without the wrapper), so it expects:
- Model: `vision_encoder.pos_embed`
- Model: `perceiver.latents`
- Model: `lang_encoder.model.layers.0.gated_cross_attn_layer.*`

Result: **716 missing keys, 716 unexpected keys** → Model uses random weights → Nonsense outputs

## The Fix

Strip the `model.` prefix from checkpoint keys before loading:

```python
# Load checkpoint
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

# CRITICAL FIX: Strip 'model.' prefix from checkpoint keys
state_dict = ckpt['model_state']
fixed_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        new_key = key[6:]  # Remove 'model.' prefix (6 characters)
        fixed_state_dict[new_key] = value
    else:
        fixed_state_dict[key] = value

# Load the fixed state dict
missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
```

## Verification

With the fix applied:
- **Missing keys: 0** (was 716) ✓
- **All model weights load correctly** ✓
- **Model ready for proper evaluation** ✓

The remaining "unexpected keys" (716) are duplicates from the checkpoint containing both `model.*` and `llm.*` versions (since `self.llm = model` in training code). These can be safely ignored.

## Files Fixed

- `eval_llama32_1b_ecg_flamingo_mimiciv_final.py` - Updated with the fix

## Files That Need the Same Fix

The following evaluation scripts have the same bug and need the same fix:
- `eval_llama32_1b_ecg_flamingo_mimiciv_v2.py` (line 147)
- `eval_llama32_1b_ecg_flamingo_mimiciv_v3.py` (line 190)

## Test Script

Run `test_checkpoint_loading.py` to verify the fix works correctly.
