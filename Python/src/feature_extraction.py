import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
def extract_time_domain_features(epoch):
    """
    EXAMPLE: Extract basic time-domain features from a single epoch.

    This is a MINIMAL example with only 3 features.
    Students must implement the remaining 13+ time-domain features.

    Works for any signal type (EEG, EOG, EMG) but students should consider
    signal-specific features for optimal performance.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        dict: A dictionary of features.
    """
    # EXAMPLE: Only 3 basic features - students must add 13+ more
    features = {
        'mean': np.mean(epoch),
        'median': np.median(epoch),
        'std': np.std(epoch),
    }

    # TODO: Students must implement remaining time-domain features:
    # Basic statistical features:
    # 4
    features['variance'] = np.var(epoch)
    # 5
    features['rms'] = np.sqrt(np.mean(epoch**2))
    # 6
    features['min'] = np.min(epoch)
    # 7
    features['max'] = np.max(epoch)
    # 8
    features['range'] = np.max(epoch) - np.min(epoch)
    # 9
    features['skewness'] = scipy.stats.skew(epoch)
    # 10
    features['kurtosis'] = scipy.stats.kurtosis(epoch)

    # Signal complexity features:
    # 11
    features['zero_crossings'] = np.sum(np.diff(np.sign(epoch)) != 0)
    # 12
    features['hjorth_activity'] = np.var(epoch)
    # 13
    features['hjorth_mobility'] = np.sqrt(np.var(np.diff(epoch)) / np.var(epoch))
    # 14
    features['hjorth_complexity'] = hjorth_complexity(epoch)

    # Signal energy and power:
    # 15
    features['total_energy'] = np.sum(epoch**2)
    # 16
    features['mean_power'] = np.mean(epoch**2)

    return features



def extract_features(data, config):
    """
    STUDENT IMPLEMENTATION AREA: Extract features based on current iteration.

    This function should handle both single-channel (old format) and
    multi-channel data (new format with 2 EEG + 2 EOG + 1 EMG channels).

    Iteration 1: 16 time-domain features per EEG channel
    Iteration 2: 31+ features (time + frequency domain) per channel
    Iteration 3: Multi-signal features (EEG + EOG + EMG)
    Iteration 4: Optimized feature set (selected subset)

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        config (module): The configuration module.

    Returns:
        np.ndarray: A 2D array of features (n_epochs, n_features).
    """
    print(f"Extracting features for iteration {config.CURRENT_ITERATION}...")

    # Detect if we have multi-channel data structure
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return extract_multi_channel_features(data, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return extract_single_channel_features(data, config)


def extract_multi_channel_features(multi_channel_data, config):
    """
    Extract features from multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.

    Students should expand this significantly!
    """
    n_epochs = multi_channel_data['eeg'].shape[0]
    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = []

        # EEG features (2 channels)
        for ch in range(multi_channel_data['eeg'].shape[1]):
            eeg_signal = multi_channel_data['eeg'][epoch_idx, ch, :]
            eeg_features = extract_time_domain_features(eeg_signal)
            epoch_features.extend(list(eeg_features.values()))

        if config.CURRENT_ITERATION >= 3:
            # Add EOG features (2 channels)
            for ch in range(multi_channel_data['eog'].shape[1]):
                eog_signal = multi_channel_data['eog'][epoch_idx, ch, :]
                eog_features = extract_eog_features(eog_signal)
                epoch_features.extend(list(eog_features.values()))

            # Add EMG features (1 channel)
            emg_signal = multi_channel_data['emg'][epoch_idx, 0, :]
            emg_features = extract_emg_features(emg_signal)
            epoch_features.extend(list(emg_features.values()))

        all_features.append(epoch_features)

    features = np.array(all_features)

    if config.CURRENT_ITERATION == 1:
        expected = 2 * 3  # 2 EEG channels × 3 features each
        print(f"Multi-channel Iteration 1: {features.shape[1]} features (target: {expected}+)")
        print("Students must implement remaining 13 time-domain features per EEG channel!")
    elif config.CURRENT_ITERATION >= 3:
        print(f"Multi-channel features extracted: {features.shape[1]} total")
        print("(2 EEG + 2 EOG + 1 EMG channels)")

    return features


def extract_single_channel_features(data, config):
    """
    Backward compatibility for single-channel data.
    """
    if config.CURRENT_ITERATION == 1:
        # Iteration 1: Time-domain features (TARGET: 16 features)
        # CURRENT: Only 3 features implemented - students must add 13 more!
        all_features = []
        for epoch_index,epoch in enumerate(data):
            features = extract_time_domain_features(epoch)
            # visualize_features(data,epoch_index)  # Visualize features for debugging
            all_features.append(list(features.values()))
        feature_names = list(extract_time_domain_features(data[0]).keys())   
        df_features = pd.DataFrame(all_features, columns=feature_names)
        scaler = RobustScaler()
        normalized_array = scaler.fit_transform(df_features)
        df_normalized = pd.DataFrame(normalized_array, columns=feature_names)
        all_features = df_normalized.values.tolist()
        features = np.array(all_features)     # 🔹 Show single-epoch visualization (first epoch by default)
        visualize_feature_distributions(features, feature_names)
        visualize_feature_trends(features, feature_names)


        print(f"WARNING: Only {features.shape[1]} features extracted, target is 16 for iteration 1")
        print("Students must implement the remaining time-domain features!")

    elif config.CURRENT_ITERATION == 2:
        # TODO: Students must implement frequency-domain features
        print("TODO: Students must implement frequency-domain feature extraction")
        print("Target: ~31 features (time + frequency domain)")
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))  # Empty features - students must implement

    elif config.CURRENT_ITERATION >= 3:
        # TODO: Students must implement multi-signal features
        print("TODO: Students should use multi-channel data format for iteration 3+")
        n_epochs = data.shape[0] if len(data.shape) > 1 else 1
        features = np.zeros((n_epochs, 0))  # Empty features - students must implement

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return features


def extract_eog_features(eog_signal):
    """
    STUDENT TODO: Extract EOG-specific features for eye movement detection.

    EOG signals are used to detect:
    - Rapid eye movements (REM sleep indicator)
    - Slow eye movements
    - Eye blinks and artifacts
    """
    features = {
        'eog_mean': np.mean(eog_signal),
        'eog_std': np.std(eog_signal),
        'eog_range': np.max(eog_signal) - np.min(eog_signal),
    }

    # TODO: Students should add:
    # - Eye movement detection features
    # - Rapid vs slow movement discrimination
    # - Cross-channel correlations (left vs right eye)

    return features


def extract_emg_features(emg_signal):
    """
    STUDENT TODO: Extract EMG-specific features for muscle tone detection.

    EMG signals are used to detect:
    - Muscle tone levels (high in wake, low in REM)
    - Muscle twitches and artifacts
    - Sleep-related muscle activity
    """
    features = {
        'emg_mean': np.mean(emg_signal),
        'emg_std': np.std(emg_signal),
        'emg_rms': np.sqrt(np.mean(emg_signal**2)),
    }

    # TODO: Students should add:
    # - High-frequency power (muscle activity indicator)
    # - Spectral edge frequency
    # - Muscle tone quantification

    return features

def hjorth_complexity(epoch):
    """
    Calculate Hjorth Complexity of a signal epoch.

    Hjorth Complexity is a measure of the shape of the signal waveform,
    indicating how the frequency content changes over time.

    Args:
        epoch (np.ndarray): A 1D array representing one epoch of signal data.

    Returns:
        float: The Hjorth Complexity value.
    """
    first_deriv = np.diff(epoch)
    second_deriv = np.diff(first_deriv)

    var_zero = np.var(epoch)
    var_d1 = np.var(first_deriv)
    var_d2 = np.var(second_deriv)

    if var_zero == 0 or var_d1 == 0:
        return 0.0

    mobility = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / mobility

    return complexity

def visualize_features(data, epoch_index, all_features=None, normalize=True):
    """
    Visualize a single epoch and its time-domain features, with optional dataset-wide normalization.

    Args:
        data (np.ndarray): 2D array, shape (n_epochs, n_samples)
        epoch_index (int): which epoch to visualize
        all_features (np.ndarray, optional): 2D array of shape (n_epochs, n_features)
            for computing global mean/std for normalization.
        normalize (bool): whether to normalize feature values for the bar chart
    """
    import numpy as np
    import matplotlib.pyplot as plt

    epoch = data[epoch_index]
    features = extract_time_domain_features(epoch)
    zero_crossings_idx = np.where(np.diff(np.sign(epoch)))[0]
    t = np.arange(len(epoch))

    feature_names = list(features.keys())
    feature_values = np.array(list(features.values()), dtype=float)

    # --- Normalization ---
    if normalize:
        if all_features is not None:
            # dataset-wide Z-score
            global_mean = np.mean(all_features, axis=0)
            global_std = np.std(all_features, axis=0)
            # prevent division by zero
            global_std[global_std == 0] = 1
            normalized_values = (feature_values - global_mean) / global_std
        else:
            # epoch-only Z-score
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            normalized_values = (feature_values - mean_val) / std_val if std_val != 0 else feature_values
    else:
        normalized_values = feature_values

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 1️⃣ Raw signal
    axes[0].plot(t, epoch, label='Epoch Signal')
    axes[0].axhline(features['mean'], color='r', linestyle='--', label=f"Mean = {features['mean']:.2e}")
    axes[0].scatter(zero_crossings_idx, epoch[zero_crossings_idx], color='orange', label='Zero Crossings', s=20)
    axes[0].set_title(f"Epoch {epoch_index} Signal with Key Features")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()

    # 2️⃣ Feature bar plot
    axes[1].bar(feature_names, normalized_values, color='skyblue')
    axes[1].set_title(
        f"Time-Domain Features of Epoch {epoch_index}" +
        (" (Normalized)" if normalize else "")
    )
    axes[1].set_ylabel("Feature Value (Z-score)" if normalize else "Feature Value")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def visualize_feature_distributions(features_array, feature_names):
    """
    Show feature distributions across all epochs.
    """
    df = pd.DataFrame(features_array, columns=feature_names)

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, palette="Blues")
    plt.title("Time-Domain Feature Distribution Across Epochs")
    plt.xticks(rotation=45)
    plt.ylabel("Feature Value")
    plt.tight_layout()
    plt.show()
    


def visualize_feature_trends(features_array, feature_names):
    """
    Show how each feature evolves across epochs.
    """
    plt.figure(figsize=(14, 6))
    for i, name in enumerate(feature_names):
        plt.plot(features_array[:, i], label=name)
    plt.title("Time-Domain Feature Trends Across Epochs")
    plt.xlabel("Epoch Index")
    plt.ylabel("Feature Value")
    plt.legend(ncol=4)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def review_outlier_epochs(data, features, feature_names, feature_key, fs=100, threshold=2.5):
    """
    Identify and visualize outlier epochs based on a selected feature.

    Args:
        data (np.ndarray): Raw signal data, shape (n_epochs, n_samples)
        features (np.ndarray): Feature matrix, shape (n_epochs, n_features)
        feature_names (list): List of feature names
        feature_key (str): Feature to use for outlier detection
        fs (int): Sampling frequency (Hz)
        threshold (float): Z-score threshold for outlier detection
    """
    # Convert to Z-scores
    feature_index = feature_names.index(feature_key)
    feature_values = features[:, feature_index]
    z_scores = (feature_values - np.mean(feature_values)) / np.std(feature_values)

    # Find outlier indices
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    print(f"Found {len(outlier_indices)} outlier epochs for feature '{feature_key}'")

    # Visualize each outlier
    for idx in outlier_indices:
        signal = data[idx]

        fig, axs = plt.subplots(2, 1, figsize=(12, 6))
        
        # Raw signal
        axs[0].plot(signal, color='blue')
        axs[0].set_title(f"Raw Signal - Epoch {idx} (Z-score: {z_scores[idx]:.2f})")
        axs[0].set_xlabel("Sample Index")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True)

        # Spectrogram
        f, t, Sxx = spectrogram(signal, fs=fs)
        axs[1].pcolormesh(t, f, Sxx, shading='gouraud')
        axs[1].set_title("Spectrogram")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Frequency (Hz)")

        plt.tight_layout()
        plt.show()



