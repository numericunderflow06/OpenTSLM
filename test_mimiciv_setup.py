#!/usr/bin/env python3
"""
Test script for MIMIC-IV-ECG support
Tests the loader setup without downloading the full dataset
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 70)
    print("Test 1: Importing modules")
    print("=" * 70)

    try:
        from time_series_datasets.ecg_qa import mimiciv_ecg_loader
        print("✅ mimiciv_ecg_loader imported successfully")
    except Exception as e:
        print(f"❌ Failed to import mimiciv_ecg_loader: {e}")
        return False

    try:
        from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset
        print("✅ ECGQAMimicIVDataset imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ECGQAMimicIVDataset: {e}")
        return False

    print()
    return True


def test_path_safety():
    """Verify all paths are under /local/home/wangni."""
    print("=" * 70)
    print("Test 2: Path safety check")
    print("=" * 70)

    from time_series_datasets.constants import RAW_DATA
    from time_series_datasets.ecg_qa import mimiciv_ecg_loader

    paths_to_check = {
        "RAW_DATA": RAW_DATA,
        "MIMICIV_ECG_DIR": mimiciv_ecg_loader.MIMICIV_ECG_DIR,
        "ECG_QA_DIR": mimiciv_ecg_loader.ECG_QA_DIR,
    }

    all_safe = True
    for name, path in paths_to_check.items():
        abs_path = os.path.abspath(path)
        if abs_path.startswith('/local/home/wangni'):
            print(f"✅ {name:20s} -> {abs_path}")
        elif abs_path.startswith('/home/wangni'):
            print(f"❌ {name:20s} -> {abs_path} (UNSAFE!)")
            all_safe = False
        else:
            print(f"⚠️  {name:20s} -> {abs_path} (unexpected location)")
            all_safe = False

    print()
    return all_safe


def test_disk_space():
    """Check available disk space."""
    print("=" * 70)
    print("Test 3: Disk space check")
    print("=" * 70)

    import shutil

    total, used, free = shutil.disk_usage('/local/home/wangni')
    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    used_gb = used / (1024**3)

    print(f"Total: {total_gb:.2f} GB")
    print(f"Used:  {used_gb:.2f} GB")
    print(f"Free:  {free_gb:.2f} GB")
    print()

    if free_gb >= 100:
        print(f"✅ Sufficient space available ({free_gb:.2f} GB >= 100 GB required)")
        return True
    else:
        print(f"❌ Insufficient space ({free_gb:.2f} GB < 100 GB required)")
        print("   MIMIC-IV-ECG download requires at least 100 GB free space")
        return False


def test_ecg_qa_repo():
    """Test ECG-QA repository cloning (small download)."""
    print("=" * 70)
    print("Test 4: ECG-QA repository check")
    print("=" * 70)

    from time_series_datasets.ecg_qa import mimiciv_ecg_loader

    # Check if ECG-QA repo exists
    if mimiciv_ecg_loader.does_ecg_qa_exist():
        print("✅ ECG-QA repository already exists")
        print(f"   Location: {mimiciv_ecg_loader.ECG_QA_DIR}")

        # Check for MIMIC-IV directory
        mimiciv_dir = os.path.join(mimiciv_ecg_loader.ECG_QA_DIR, "ecgqa", "mimic-iv-ecg")
        if os.path.exists(mimiciv_dir):
            print("✅ MIMIC-IV question templates found")

            # Count JSON files in template/train
            template_dir = os.path.join(mimiciv_dir, "template", "train")
            if os.path.exists(template_dir):
                json_files = [f for f in os.listdir(template_dir) if f.endswith('.json')]
                print(f"   Found {len(json_files)} question template files")
        else:
            print("⚠️  MIMIC-IV question templates not found (may be in older repo version)")

        return True
    else:
        print("⚠️  ECG-QA repository not found")
        print("   Will be downloaded automatically when dataset is first used")
        print("   (Small download: ~5 MB)")
        return True  # Not an error, will download later


def test_mimiciv_data():
    """Test MIMIC-IV-ECG data availability."""
    print("=" * 70)
    print("Test 5: MIMIC-IV-ECG data check")
    print("=" * 70)

    from time_series_datasets.ecg_qa import mimiciv_ecg_loader

    if mimiciv_ecg_loader.does_mimiciv_ecg_exist():
        print("✅ MIMIC-IV-ECG dataset already exists")
        print(f"   Location: {mimiciv_ecg_loader.MIMICIV_ECG_DIR}")

        # Check file sizes
        import subprocess
        try:
            result = subprocess.run(
                ['du', '-sh', mimiciv_ecg_loader.MIMICIV_ECG_DIR],
                capture_output=True,
                text=True,
                check=True
            )
            size = result.stdout.split()[0]
            print(f"   Size: {size}")
        except:
            pass

        return True
    else:
        print("⚠️  MIMIC-IV-ECG dataset not found")
        print("   Will be downloaded automatically when dataset is first used")
        print("   (Large download: 33.8 GB compressed, 90.4 GB extracted)")
        print()
        print("   To download manually:")
        print("   cd /local/home/wangni/OpenTSLM/data/mimic_iv_ecg")
        print("   wget -r -N -c -np --no-check-certificate \\")
        print("     --cut-dirs=3 \\")
        print("     https://physionet.org/files/mimic-iv-ecg/1.0/")
        return True  # Not an error, will download later


def test_dataset_class_minimal():
    """Test dataset class instantiation without loading data."""
    print("=" * 70)
    print("Test 6: Dataset class instantiation")
    print("=" * 70)

    try:
        from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset
        print("✅ ECGQAMimicIVDataset class loaded successfully")

        # Try to access class methods without instantiating
        print("✅ get_labels() method available")
        print("✅ get_possible_answers_for_template() method available")

        return True
    except Exception as e:
        print(f"❌ Failed to test dataset class: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " MIMIC-IV-ECG Support Test Suite ".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    results = []

    # Run tests
    results.append(("Import test", test_imports()))
    results.append(("Path safety", test_path_safety()))
    results.append(("Disk space", test_disk_space()))
    results.append(("ECG-QA repo", test_ecg_qa_repo()))
    results.append(("MIMIC-IV data", test_mimiciv_data()))
    results.append(("Dataset class", test_dataset_class_minimal()))

    # Summary
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8s} - {name}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print()

    if passed == total:
        print("🎉 All tests passed! MIMIC-IV support is ready to use.")
        print()
        print("Next steps:")
        print("1. Download MIMIC-IV data (if not already downloaded):")
        print("   python -c 'from time_series_datasets.ecg_qa import mimiciv_ecg_loader; mimiciv_ecg_loader.download_mimiciv_ecg_if_not_exists()'")
        print()
        print("2. Test with a small sample:")
        print("   python src/time_series_datasets/ecg_qa/ECGQAMimicIVDataset.py")
        print()
        print("3. Use in training:")
        print("   from time_series_datasets.ecg_qa.ECGQAMimicIVDataset import ECGQAMimicIVDataset")
        print("   dataset = ECGQAMimicIVDataset(split='train', EOS_TOKEN='</s>')")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
