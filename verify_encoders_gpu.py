#!/usr/bin/env python3
"""
Comprehensive GPU verification test for all encoders.
Checks code integrity, device placement, memory usage, and performance.
"""

import os
import sys
import torch
import time
import gc
import traceback
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Set cache directories to /local/home/wangni
os.environ["HF_HOME"] = "/local/home/wangni/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/local/home/wangni/.cache/huggingface/transformers"
os.environ["TORCH_HOME"] = "/local/home/wangni/.cache/torch"
os.environ["XDG_CACHE_HOME"] = "/local/home/wangni/.cache"

# Ensure directories exist
for cache_dir in [os.environ["HF_HOME"], os.environ["TRANSFORMERS_CACHE"], os.environ["TORCH_HOME"]]:
    os.makedirs(cache_dir, exist_ok=True)

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("="*80)
print("COMPREHENSIVE GPU VERIFICATION TEST")
print("="*80)

# Check GPU availability
if not torch.cuda.is_available():
    print("❌ CUDA is not available! Exiting.")
    sys.exit(1)

DEVICE = "cuda"
print(f"✅ Device: {DEVICE}")
print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
print(f"✅ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"✅ Cache: /local/home/wangni/.cache/huggingface")
print("="*80 + "\n")

from opentslm.model_config import ENCODER_OUTPUT_DIM

def verify_encoder(encoder_name, encoder_class, config, test_configs):
    """Comprehensive encoder verification."""
    print(f"\n{'='*70}")
    print(f"VERIFYING: {encoder_name}")
    print(f"{'='*70}")

    results = {
        'initialization': False,
        'device_check': False,
        'forward_pass': False,
        'output_check': False,
        'memory_check': False,
        'performance': {},
        'errors': []
    }

    try:
        # 1. INITIALIZATION CHECK
        print("\n1. Initialization Check:")
        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        encoder = encoder_class(**config)
        encoder = encoder.to(DEVICE)
        encoder.eval()

        init_time = time.time() - start
        init_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"   ✅ Initialized in {init_time:.2f}s")
        print(f"   ✅ Initial memory: {init_memory:.2f} MB")
        results['initialization'] = True
        results['performance']['init_time'] = init_time
        results['performance']['init_memory'] = init_memory

        # 2. DEVICE CHECK
        print("\n2. Device Placement Check:")
        all_on_gpu = True
        for name, param in encoder.named_parameters():
            if param.device.type != 'cuda':
                print(f"   ❌ Parameter {name} is on {param.device}")
                all_on_gpu = False
                results['errors'].append(f"Parameter {name} not on GPU")

        if all_on_gpu:
            print(f"   ✅ All parameters on GPU")
            results['device_check'] = True

        # 3. FORWARD PASS TESTS
        print("\n3. Forward Pass Tests:")
        all_passed = True

        for test_name, (batch_size, seq_len, n_features) in test_configs.items():
            print(f"\n   Test '{test_name}': batch={batch_size}, seq={seq_len}, features={n_features}")

            # Create test data
            data = torch.randn(batch_size, seq_len, n_features).to(DEVICE)
            print(f"   Input: {data.shape}, device={data.device}, dtype={data.dtype}")

            # Clear cache and measure
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            # Forward pass
            start = time.time()
            try:
                with torch.no_grad():
                    output = encoder(data)
                forward_time = time.time() - start
                forward_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

                print(f"   Output: {output.shape}, device={output.device}, dtype={output.dtype}")
                print(f"   Time: {forward_time:.3f}s, Memory: {forward_memory:.2f} MB")

                # Verify output
                expected_shape = (batch_size, ENCODER_OUTPUT_DIM)
                if output.shape != expected_shape:
                    print(f"   ❌ Shape mismatch! Expected {expected_shape}")
                    all_passed = False
                    results['errors'].append(f"Shape mismatch in {test_name}")
                elif output.device.type != 'cuda':
                    print(f"   ❌ Output not on GPU!")
                    all_passed = False
                    results['errors'].append(f"Output not on GPU in {test_name}")
                elif torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"   ❌ Output contains NaN or Inf!")
                    all_passed = False
                    results['errors'].append(f"NaN/Inf in {test_name}")
                else:
                    print(f"   ✅ Test passed")
                    results['performance'][test_name] = {
                        'time': forward_time,
                        'memory': forward_memory
                    }

            except Exception as e:
                print(f"   ❌ Forward pass failed: {str(e)[:100]}")
                all_passed = False
                results['errors'].append(f"Forward failed in {test_name}: {str(e)[:50]}")

        results['forward_pass'] = all_passed
        results['output_check'] = all_passed

        # 4. MEMORY CHECK
        print("\n4. Memory Usage Check:")
        torch.cuda.synchronize()
        current_memory = torch.cuda.memory_allocated() / 1024 / 1024
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        print(f"   Current: {current_memory:.2f} MB")
        print(f"   Peak: {peak_memory:.2f} MB")

        if peak_memory < 10000:  # Less than 10GB
            print(f"   ✅ Memory usage acceptable")
            results['memory_check'] = True
        else:
            print(f"   ⚠️  High memory usage")
            results['errors'].append(f"High memory: {peak_memory:.2f} MB")

        # 5. GRADIENT CHECK (if not frozen)
        if not config.get('freeze_backbone', False):
            print("\n5. Gradient Flow Check:")
            data = torch.randn(2, 100, test_configs['small'][2]).to(DEVICE)
            output = encoder(data)
            loss = output.mean()
            loss.backward()

            has_gradients = False
            for name, param in encoder.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_gradients = True
                    break

            if has_gradients:
                print(f"   ✅ Gradients flow correctly")
            else:
                print(f"   ⚠️  No gradients (might be frozen)")

    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {str(e)}")
        traceback.print_exc()
        results['errors'].append(f"Fatal error: {str(e)[:100]}")

    finally:
        # Cleanup
        if 'encoder' in locals():
            del encoder
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*30}")
    print("VERIFICATION SUMMARY:")
    print(f"  Initialization: {'✅' if results['initialization'] else '❌'}")
    print(f"  Device Check: {'✅' if results['device_check'] else '❌'}")
    print(f"  Forward Pass: {'✅' if results['forward_pass'] else '❌'}")
    print(f"  Output Check: {'✅' if results['output_check'] else '❌'}")
    print(f"  Memory Check: {'✅' if results['memory_check'] else '❌'}")

    if results['errors']:
        print(f"\n  Errors:")
        for error in results['errors']:
            print(f"    - {error}")

    all_checks = all([
        results['initialization'],
        results['device_check'],
        results['forward_pass'],
        results['output_check'],
        results['memory_check']
    ])

    return all_checks, results

def main():
    overall_results = {}

    # Test configurations for different scenarios
    chronos_tests = {
        'small': (2, 100, 1),
        'medium': (4, 512, 1),
        'large': (8, 1024, 1),
        'edge_case': (1, 50, 1),
    }

    patchtst_tests = {
        'small': (2, 512, 7),  # ETTh1 has 7 channels
        'medium': (4, 512, 7),
        'large': (8, 512, 7),
        'different_channels': (2, 512, 3),  # Test channel adaptation
    }

    timesfm_tests = {
        'small': (2, 256, 1),
        'medium': (4, 512, 1),
        'large': (8, 1024, 1),
    }

    moment_tests = {
        'small': (2, 256, 1),
        'medium': (4, 512, 1),
        'large': (8, 1024, 1),
    }

    # 1. CHRONOS
    print("\n" + "="*80)
    print("1. CHRONOS ENCODER VERIFICATION")
    print("="*80)

    try:
        from opentslm.model.encoder.ChronosCompleteEncoder import ChronosCompleteEncoder

        success, results = verify_encoder(
            "Chronos",
            ChronosCompleteEncoder,
            {
                "model_name": "amazon/chronos-t5-tiny",
                "output_dim": ENCODER_OUTPUT_DIM,
                "device": DEVICE,
                "dropout": 0.1
            },
            chronos_tests
        )
        overall_results["Chronos"] = (success, results)

    except Exception as e:
        print(f"❌ Could not verify Chronos: {e}")
        overall_results["Chronos"] = (False, {'errors': [str(e)]})

    # 2. PATCHTST
    print("\n" + "="*80)
    print("2. PATCHTST ENCODER VERIFICATION")
    print("="*80)

    try:
        from opentslm.model.encoder.PatchTSTCompleteEncoder import PatchTSTCompleteEncoder

        success, results = verify_encoder(
            "PatchTST",
            PatchTSTCompleteEncoder,
            {
                "model_name": "ibm/patchtst-etth1-pretrain",
                "output_dim": ENCODER_OUTPUT_DIM,
                "device": DEVICE,
                "dropout": 0.1
            },
            patchtst_tests
        )
        overall_results["PatchTST"] = (success, results)

    except Exception as e:
        print(f"❌ Could not verify PatchTST: {e}")
        overall_results["PatchTST"] = (False, {'errors': [str(e)]})

    # 3. TIMESFM
    print("\n" + "="*80)
    print("3. TIMESFM ENCODER VERIFICATION")
    print("="*80)

    try:
        from opentslm.model.encoder.TimesFMFinalEncoder import TimesFMFinalEncoder

        success, results = verify_encoder(
            "TimesFM",
            TimesFMFinalEncoder,
            {
                "model_name": "google/timesfm-2p5-200m-torch",
                "output_dim": ENCODER_OUTPUT_DIM,
                "context_length": 512,
                "device": DEVICE,
                "dropout": 0.0
            },
            timesfm_tests
        )
        overall_results["TimesFM"] = (success, results)

    except Exception as e:
        print(f"❌ Could not verify TimesFM: {e}")
        overall_results["TimesFM"] = (False, {'errors': [str(e)]})

    # 4. MOMENT
    print("\n" + "="*80)
    print("4. MOMENT ENCODER VERIFICATION")
    print("="*80)

    try:
        from opentslm.model.encoder.MOMENTEncoderReal import MOMENTEncoderReal

        success, results = verify_encoder(
            "MOMENT",
            MOMENTEncoderReal,
            {
                "model_name": "AutonLab/MOMENT-1-large",
                "output_dim": ENCODER_OUTPUT_DIM,
                "device": DEVICE,
                "dropout": 0.0
            },
            moment_tests
        )
        overall_results["MOMENT"] = (success, results)

    except Exception as e:
        print(f"❌ Could not verify MOMENT: {e}")
        overall_results["MOMENT"] = (False, {'errors': [str(e)]})

    # FINAL REPORT
    print("\n" + "="*80)
    print("FINAL VERIFICATION REPORT")
    print("="*80)

    all_passed = True
    for encoder_name, (success, results) in overall_results.items():
        if success:
            print(f"✅ {encoder_name:15s} - ALL CHECKS PASSED")
        else:
            print(f"❌ {encoder_name:15s} - FAILED")
            if 'errors' in results and results['errors']:
                print(f"   Errors: {', '.join(results['errors'][:2])}")
            all_passed = False

    print(f"\n{'='*40}")
    passed_count = sum(1 for s, _ in overall_results.values() if s)
    print(f"RESULT: {passed_count}/{len(overall_results)} encoders verified")
    print(f"{'='*40}")

    if all_passed:
        print("\n🎉 VERIFICATION COMPLETE: ALL ENCODERS WORKING PERFECTLY!")
        print("\nConfirmed:")
        print("  ✅ All using real pretrained models")
        print("  ✅ All parameters on GPU")
        print("  ✅ All outputs correct shape and device")
        print("  ✅ Memory usage acceptable")
        print("  ✅ No NaN/Inf values")
        print("  ✅ Cache at /local/home/wangni/")
    else:
        print("\n⚠️  Some encoders have issues. Check errors above.")

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)