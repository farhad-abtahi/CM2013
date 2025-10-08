"""
Test script for data loader functionality.

This script tests the complete data loading pipeline with real data.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_training_data, load_all_training_data
from xml_parser import parse_xml_annotations

def test_xml_parser():
    """Test XML parsing."""
    print("=" * 60)
    print("TEST 1: XML Parser")
    print("=" * 60)

    xml_file = '../data/training/R1.xml'

    if not os.path.exists(xml_file):
        print(f"SKIP: {xml_file} not found")
        return

    result = parse_xml_annotations(xml_file)

    print(f"✓ Parsed XML file: {xml_file}")
    print(f"  - Epoch length: {result['epoch_length']} seconds")
    print(f"  - Stage events: {len(result['stages'])}")
    print(f"  - Total events: {len(result['events'])}")

    # Calculate duration
    if result['stages']:
        total_duration = sum(s['duration'] for s in result['stages'])
        print(f"  - Total duration: {total_duration/3600:.2f} hours")

    print()


def test_single_recording():
    """Test loading a single EDF + XML pair."""
    print("=" * 60)
    print("TEST 2: Single Recording Load")
    print("=" * 60)

    edf_file = '../data/training/R1.edf'
    xml_file = '../data/training/R1.xml'

    if not os.path.exists(edf_file) or not os.path.exists(xml_file):
        print(f"SKIP: Files not found")
        return

    data, labels, info = load_training_data(edf_file, xml_file)

    print(f"\n✓ Successfully loaded R1")
    print(f"  - Total epochs: {len(labels)}")
    print(f"  - Duration: {len(labels)*30/3600:.2f} hours")

    # Verify data structure
    print(f"\n  Data structure:")
    for signal_type in data.keys():
        shape = data[signal_type].shape
        print(f"    {signal_type.upper()}: {shape}")

    # Verify label distribution
    import numpy as np
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    print(f"\n  Label distribution:")
    for stage in range(5):
        count = np.sum(labels == stage)
        if count > 0:
            pct = (count / len(labels)) * 100
            print(f"    {stage_names[stage]}: {count} epochs ({pct:.1f}%)")

    print()


def test_multiple_recordings():
    """Test loading all training recordings."""
    print("=" * 60)
    print("TEST 3: Multiple Recordings Load")
    print("=" * 60)

    training_dir = '../data/training/'

    if not os.path.exists(training_dir):
        print(f"SKIP: {training_dir} not found")
        return

    # Count available EDF files
    import glob
    edf_files = glob.glob(os.path.join(training_dir, '*.edf'))
    print(f"Found {len(edf_files)} EDF files in {training_dir}")

    if len(edf_files) == 0:
        print("SKIP: No EDF files found")
        return

    # Load all recordings
    print("\nLoading all recordings...")
    data, labels, record_ids, info = load_all_training_data(training_dir)

    print(f"\n✓ Successfully loaded all recordings")

    # Verify combined data
    import numpy as np
    print(f"\n  Combined data:")
    print(f"    Total epochs: {len(labels)}")
    print(f"    Total duration: {len(labels)*30/3600:.2f} hours")
    print(f"    Unique recordings: {len(np.unique(record_ids))}")

    for signal_type in data.keys():
        shape = data[signal_type].shape
        print(f"    {signal_type.upper()}: {shape}")

    # Verify record ID tracking
    print(f"\n  Recording distribution:")
    for record_id in sorted(np.unique(record_ids)):
        count = np.sum(record_ids == record_id)
        print(f"    {record_id}: {count} epochs")

    print()


def test_loso_compatibility():
    """Test that data is compatible with LOSO cross-validation."""
    print("=" * 60)
    print("TEST 4: LOSO Compatibility")
    print("=" * 60)

    training_dir = '../data/training/'

    if not os.path.exists(training_dir):
        print(f"SKIP: {training_dir} not found")
        return

    # Load data
    import glob
    edf_files = glob.glob(os.path.join(training_dir, '*.edf'))

    if len(edf_files) < 2:
        print("SKIP: Need at least 2 recordings for LOSO test")
        return

    data, labels, record_ids, info = load_all_training_data(training_dir)

    # Test LOSO split
    try:
        from sklearn.model_selection import LeaveOneGroupOut
        import numpy as np

        logo = LeaveOneGroupOut()
        n_splits = logo.get_n_splits(groups=record_ids)

        print(f"✓ LOSO split created successfully")
        print(f"  - Number of folds: {n_splits}")

        # Verify first split
        for train_idx, test_idx in logo.split(np.zeros(len(labels)), labels, groups=record_ids):
            test_subject = np.unique(record_ids[test_idx])[0]
            n_train = len(train_idx)
            n_test = len(test_idx)

            print(f"\n  Example fold (test on {test_subject}):")
            print(f"    Training epochs: {n_train}")
            print(f"    Test epochs: {n_test}")
            break

    except ImportError:
        print("SKIP: scikit-learn not installed")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("DATA LOADER TEST SUITE")
    print("=" * 60)
    print()

    test_xml_parser()
    test_single_recording()
    test_multiple_recordings()
    test_loso_compatibility()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
