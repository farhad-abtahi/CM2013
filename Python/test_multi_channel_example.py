#!/usr/bin/env python3
"""
Sleep Scoring - Multi-Channel Feature Extraction Example

This script demonstrates proper handling of multi-channel EDF files
with R1.edf and R1.xml as reference examples.

STUDENTS: This is an EXAMPLE showing multi-channel feature extraction.
You should understand this approach and adapt it for your pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from data_loader import load_training_data

def main():
    print("=== Multi-Channel Feature Extraction Example ===")

    # Example files - students should adapt to their dataset
    edf_file = os.path.join(config.SAMPLE_DIR, "R1.edf")
    xml_file = os.path.join(config.SAMPLE_DIR, "R1.xml")

    print(f"\nTarget files: {edf_file}, {xml_file}")

    try:
        # Load multi-channel data
        multi_channel_data, labels, channel_info = load_training_data(edf_file, xml_file)

        print("\n--- Multi-Channel Data Structure ---")
        print(f"EEG channels: {channel_info['eeg_names']} at {channel_info['eeg_fs']} Hz")
        print(f"EOG channels: {channel_info['eog_names']} at {channel_info['eog_fs']} Hz")
        print(f"EMG channels: {channel_info['emg_names']} at {channel_info['emg_fs']} Hz")

        print(f"\nData shapes:")
        print(f"  EEG: {multi_channel_data['eeg'].shape} (epochs, channels, samples)")
        print(f"  EOG: {multi_channel_data['eog'].shape}")
        print(f"  EMG: {multi_channel_data['emg'].shape}")
        print(f"Labels: {labels.shape}")

        # Demonstrate multi-channel feature extraction
        demonstrate_multi_channel_features(multi_channel_data, labels, channel_info)

        # Create visualizations
        create_multi_channel_visualization(multi_channel_data, labels, channel_info)

    except Exception as e:
        print(f"Note: Using dummy data structure (real files not found): {e}")
        demonstrate_expected_structure()

def demonstrate_multi_channel_features(data, labels, channel_info):
    """Show how to extract features from multi-channel data."""

    print("\n--- Multi-Channel Feature Extraction Example ---")

    n_epochs = data['eeg'].shape[0]
    epoch_example = 50  # Show features for epoch 50

    if epoch_example >= n_epochs:
        epoch_example = 0

    print(f"Extracting features for epoch {epoch_example}:")

    # EEG features (2 channels)
    eeg_epoch = data['eeg'][epoch_example]  # Shape: (2, samples)
    eeg_features = extract_eeg_features(eeg_epoch, channel_info['eeg_fs'])
    print(f"EEG features: {len(eeg_features)} total")

    # EOG features (2 channels)
    eog_epoch = data['eog'][epoch_example]  # Shape: (2, samples)
    eog_features = extract_eog_features(eog_epoch, channel_info['eog_fs'])
    print(f"EOG features: {len(eog_features)} total")

    # EMG features (1 channel)
    emg_epoch = data['emg'][epoch_example]  # Shape: (1, samples)
    emg_features = extract_emg_features(emg_epoch, channel_info['emg_fs'])
    print(f"EMG features: {len(emg_features)} total")

    # Combined feature vector
    all_features = {**eeg_features, **eog_features, **emg_features}
    print(f"\nTotal combined features: {len(all_features)}")

    # Show sleep stage for this epoch
    stage_names = ['REM', 'N3', 'N2', 'N1', 'Wake']
    print(f"Sleep stage for epoch {epoch_example}: {stage_names[labels[epoch_example]]}")

    return all_features

def extract_eeg_features(eeg_data, fs):
    """
    Extract features from 2 EEG channels.

    Students should expand this significantly for their implementation.
    """
    features = {}

    for ch in range(eeg_data.shape[0]):
        signal = eeg_data[ch, :]
        ch_name = f"eeg_ch{ch+1}"

        # Basic time-domain features (students should add more)
        features[f"{ch_name}_mean"] = np.mean(signal)
        features[f"{ch_name}_std"] = np.std(signal)
        features[f"{ch_name}_var"] = np.var(signal)
        features[f"{ch_name}_rms"] = np.sqrt(np.mean(signal**2))
        features[f"{ch_name}_range"] = np.max(signal) - np.min(signal)

        # Basic frequency-domain features (students should expand)
        fft_signal = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/fs)
        power_spectrum = np.abs(fft_signal)**2

        # Band power features
        delta_power = get_band_power(power_spectrum, freqs, 0.5, 4)
        theta_power = get_band_power(power_spectrum, freqs, 4, 8)
        alpha_power = get_band_power(power_spectrum, freqs, 8, 13)
        beta_power = get_band_power(power_spectrum, freqs, 13, 30)

        features[f"{ch_name}_delta_power"] = delta_power
        features[f"{ch_name}_theta_power"] = theta_power
        features[f"{ch_name}_alpha_power"] = alpha_power
        features[f"{ch_name}_beta_power"] = beta_power

        # TODO: Students should add:
        # - Hjorth parameters (activity, mobility, complexity)
        # - Spectral entropy
        # - Zero crossing rate
        # - More sophisticated frequency features

    return features

def extract_eog_features(eog_data, fs):
    """Extract features from 2 EOG channels (eye movement detection)."""
    features = {}

    for ch in range(eog_data.shape[0]):
        signal = eog_data[ch, :]
        ch_name = f"eog_ch{ch+1}"

        # EOG-specific features for eye movement detection
        features[f"{ch_name}_mean"] = np.mean(signal)
        features[f"{ch_name}_std"] = np.std(signal)
        features[f"{ch_name}_max_amplitude"] = np.max(np.abs(signal))

        # Eye movement indicators (students should expand)
        # Rapid eye movements are characteristic of REM sleep
        signal_diff = np.diff(signal)
        features[f"{ch_name}_movement_energy"] = np.sum(signal_diff**2)

        # TODO: Students should add:
        # - Slow eye movement detection
        # - Rapid eye movement detection
        # - Blink artifact detection

    # Cross-channel EOG features (between left and right eye)
    if eog_data.shape[0] >= 2:
        correlation = np.corrcoef(eog_data[0, :], eog_data[1, :])[0, 1]
        features["eog_lr_correlation"] = correlation

    return features

def extract_emg_features(emg_data, fs):
    """Extract features from EMG channel (muscle tone detection)."""
    features = {}

    signal = emg_data[0, :]  # Single EMG channel

    # EMG-specific features for muscle tone
    features["emg_mean"] = np.mean(signal)
    features["emg_std"] = np.std(signal)
    features["emg_rms"] = np.sqrt(np.mean(signal**2))
    features["emg_energy"] = np.sum(signal**2)

    # Muscle tone indicators
    # High-frequency content indicates muscle activity
    fft_signal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1/fs)
    power_spectrum = np.abs(fft_signal)**2

    # EMG is typically high-frequency (20-500 Hz)
    hf_power = get_band_power(power_spectrum, freqs, 20, min(250, fs/2))
    features["emg_hf_power"] = hf_power

    # TODO: Students should add:
    # - Spectral edge frequency
    # - Mean frequency
    # - Median frequency
    # - Muscle activity detection algorithms

    return features

def get_band_power(power_spectrum, freqs, low_freq, high_freq):
    """Calculate power in a specific frequency band."""
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    return np.sum(power_spectrum[mask])

def create_multi_channel_visualization(data, labels, channel_info):
    """Create visualization showing multi-channel signals and hypnogram."""

    print("\n--- Creating Multi-Channel Visualization ---")

    # Plot one 30-second epoch from all channels
    epoch_num = 100
    if epoch_num >= data['eeg'].shape[0]:
        epoch_num = 0

    fig, axes = plt.subplots(5, 1, figsize=(12, 10))

    # Time axis (30 seconds)
    t_eeg = np.arange(data['eeg'].shape[2]) / channel_info['eeg_fs']
    t_emg = np.arange(data['emg'].shape[2]) / channel_info['emg_fs']

    # EEG channels
    for ch in range(2):
        signal = data['eeg'][epoch_num, ch, :]
        axes[ch].plot(t_eeg, signal, 'b-', linewidth=0.8)
        axes[ch].set_ylabel(f"{channel_info['eeg_names'][ch]}\n(µV)")
        axes[ch].grid(True, alpha=0.3)
        axes[ch].set_xlim(0, 30)

    # EOG channels
    for ch in range(2):
        signal = data['eog'][epoch_num, ch, :]
        axes[ch+2].plot(t_eeg, signal, 'r-', linewidth=0.8)
        axes[ch+2].set_ylabel(f"{channel_info['eog_names'][ch]}\n(µV)")
        axes[ch+2].grid(True, alpha=0.3)
        axes[ch+2].set_xlim(0, 30)

    # EMG channel
    signal = data['emg'][epoch_num, 0, :]
    axes[4].plot(t_emg, signal, 'g-', linewidth=0.8)
    axes[4].set_ylabel(f"{channel_info['emg_names'][0]}\n(µV)")
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xlim(0, 30)
    axes[4].set_xlabel('Time (seconds)')

    stage_names = ['REM', 'N3', 'N2', 'N1', 'Wake']
    stage_name = stage_names[labels[epoch_num]]
    fig.suptitle(f'Multi-Channel Signals - Epoch {epoch_num} ({stage_name})')
    plt.tight_layout()
    plt.show()

    # Create hypnogram
    create_hypnogram(labels)

def create_hypnogram(labels):
    """Create a hypnogram showing sleep stage progression."""

    plt.figure(figsize=(12, 4))

    # Convert epochs to time in hours
    time_hours = np.arange(len(labels)) * 30 / 3600  # 30-second epochs to hours

    plt.plot(time_hours, labels, 'k-', linewidth=2)
    plt.ylim(-0.5, 4.5)
    plt.yticks(range(5), ['REM', 'N3', 'N2', 'N1', 'Wake'])
    plt.xlabel('Time (hours)')
    plt.ylabel('Sleep Stage')
    plt.title('Hypnogram - Sleep Stage Progression')
    plt.grid(True, alpha=0.3)

    # Add sleep statistics
    stage_names = ['REM', 'N3', 'N2', 'N1', 'Wake']
    stage_counts = [np.sum(labels == i) for i in range(5)]
    total_epochs = len(labels)

    stats_text = "Sleep Statistics:\n"
    for i, (name, count) in enumerate(zip(stage_names, stage_counts)):
        pct = count / total_epochs * 100
        stats_text += f"{name}: {pct:.1f}%\n"

    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

def demonstrate_expected_structure():
    """Show expected data structure when real files aren't available."""

    print("\n--- Expected Multi-Channel Structure ---")
    print("When you implement real EDF/XML loading, you should return:")
    print()
    print("multi_channel_data = {")
    print("    'eeg': np.array(shape=(n_epochs, 2, samples_per_epoch)),  # 2 EEG channels")
    print("    'eog': np.array(shape=(n_epochs, 2, samples_per_epoch)),  # 2 EOG channels")
    print("    'emg': np.array(shape=(n_epochs, 1, samples_per_epoch)),  # 1 EMG channel")
    print("}")
    print()
    print("channel_info = {")
    print("    'eeg_names': ['C3-A2', 'C4-A1'],")
    print("    'eeg_fs': 256,")
    print("    'eog_names': ['LOC-A2', 'ROC-A1'],")
    print("    'eog_fs': 256,")
    print("    'emg_names': ['Chin1-Chin2'],")
    print("    'emg_fs': 512")
    print("}")
    print()
    print("labels = np.array(shape=(n_epochs,))  # Sleep stages 0-4")

def print_student_guidance():
    """Print implementation guidance for students."""

    print("\n" + "="*60)
    print("STUDENT IMPLEMENTATION GUIDANCE")
    print("="*60)
    print()
    print("1. MULTI-CHANNEL DATA LOADING:")
    print("   - Use MNE or edfread to load R1.edf")
    print("   - Identify channels by name (case-insensitive search)")
    print("   - Handle different sampling rates per channel type")
    print("   - Parse R1.xml for sleep stage annotations")
    print()
    print("2. CHANNEL IDENTIFICATION:")
    print("   - EEG: Look for 'C3', 'C4' in channel names")
    print("   - EOG: Look for 'EOG', 'LOC', 'ROC' in channel names")
    print("   - EMG: Look for 'EMG', 'Chin' in channel names")
    print()
    print("3. FEATURE EXTRACTION EXPANSION:")
    print("   - Implement 16+ time-domain features per channel")
    print("   - Add frequency-domain features (band powers)")
    print("   - Include cross-channel features (correlations)")
    print("   - Consider signal-specific features (eye movements, muscle tone)")
    print()
    print("4. SAMPLING RATE HANDLING:")
    print("   - EEG/EOG typically 256 Hz")
    print("   - EMG often 512 Hz (higher for muscle activity)")
    print("   - Resample to common rate or handle natively")
    print()
    print("5. PIPELINE INTEGRATION:")
    print("   - Modify preprocessing for multi-channel data")
    print("   - Update feature extraction to handle multiple channels")
    print("   - Consider channel-specific vs combined features")
    print()

if __name__ == "__main__":
    main()
    print_student_guidance()