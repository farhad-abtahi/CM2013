from typing import Any
import numpy as np
import scipy
from scipy.stats import entropy #add by Sherry
from mne_features.univariate import compute_hjorth_complexity
from spectrum import pburg #add by Sherry
import pywt # added
from scipy.stats import skew, kurtosis # added

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
        'variance': np.var(epoch),
        'rms':np.sqrt(np.mean(epoch**2)),
        'min':np.min(epoch),
        'max': np.max(epoch),
        'range': np.max(epoch) - np.min(epoch),
        'skewness': scipy.stats.skew(epoch),
        'kurtosis': scipy.stats.kurtosis(epoch),
        'zero_crossings': np.sum(np.diff(np.sign(epoch)) != 0),
        'hjorth_activity': np.var(epoch),
        'hjorth_mobility': np.sqrt(np.var(np.diff(epoch)) / np.var(epoch)),
        'hjorth_complexity': compute_hjorth_complexity(epoch),
        'total_energy': np.sum(epoch**2),
        'mean_power': np.mean(epoch**2)
    }
    
    return features

def extract_frequency_domain_features(epoch, fs, AR_method, Welch_method, wavelet_method):

    features = {}
    AR_features = AR_method(epoch,fs)
    features.update(AR_features)
    Welch_features = Welch_method(epoch,fs)
    features.update(Welch_features)
    wavelet_features = wavelet_method(epoch,fs)
    features.update(wavelet_features)
    
    return features

def AR_method(epoch, fs):
    p=pburg(epoch, order=16, sampling=fs, criteria= 'AIC', NFFT=4096)
    psd=np.array(p.psd)
    freqs=np.array(p.frequencies())

    band={
        'delta': (0.5,4),
        'theta': (4,8),
        'alpha': (8,13),
        'beta':(13,30),
        'gamma': (30,50)
    }

    AR_features={}
    band_powers = {}

    for band_name, (low_freq, high_freq) in band.items():
        idx_band = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
        band_powers[f'{band_name}_power'] = band_power

    idx_total = np.logical_and(freqs >= 0.5, freqs < 50)
    total_power = np.trapezoid(psd[idx_total], freqs[idx_total])

    AR_features['rel_delta']=band_powers['delta_power']/(total_power+ 1e-10)
    AR_features['rel_theta']=band_powers['theta_power']/(total_power+ 1e-10)
    AR_features['rel_alpha']=band_powers['alpha_power']/(total_power+ 1e-10)
    AR_features['rel_beta']=band_powers['beta_power']/(total_power+ 1e-10)
    AR_features['rel_gamma']=band_powers['gamma_power']/(total_power+ 1e-10)

    AR_features['delta_alpha_ratio'] = band_powers['delta_power'] / (band_powers['alpha_power'] + 1e-10)
    AR_features['theta_beta_ratio'] = band_powers['theta_power'] / (band_powers['beta_power'] + 1e-10)
    AR_features['slow_fast_ratio'] = (band_powers['delta_power'] + band_powers['theta_power']) / (band_powers['alpha_power'] +band_powers['beta_power'] + 1e-10)

    cumulative_power = np.cumsum(psd)
    total_power_sef = cumulative_power[-1]
    threshold = 0.95 * total_power_sef
    idx_threshold = np.where(cumulative_power >= threshold)[0]
    if len(idx_threshold) > 0:
        sef = freqs[idx_threshold[0]]
    else:
        sef = freqs[-1]
    AR_features['edge_freq']=sef

    AR_features['ar_peak_freq'] = freqs[idx_total][np.argmax(psd[idx_total])]

    
    psd_norm = psd / (np.sum(psd)+ 1e-10)
    psd_norm = psd_norm[psd_norm > 0]
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    entropy_norm = entropy / (np.log2(len(psd_norm))+ 1e-10)

    AR_features['entropy']=entropy_norm
    
    return AR_features

def Welch_method(epoch,fs):
    Welch_features={}
    return Welch_features

# level = np.ceil(np.log2(125 / (2*0.5))) # ~= 6.97 so we can choose 7

def wavelet_method(epoch, fs, wavelet='db4', level=7):
    # this function is to extract wavelet based features from a EEG (epoch).
    # decomposiiton level = 5
    # feaures to be extract per level: energy, relative energy, entropy, mean, standard deviation, 
    # skewness and kurtosis

    coeffs = pywt.wavedec(epoch, wavelet, level=level)
    wavelet_features = {}
    total_energy = sum(np.sum(c ** 2) for c in coeffs)

    for i, coeff in enumerate(coeffs):
        band_name = f'wavelet_L{i}'
        energy = np.sum(coeff ** 2)
        rel_energy = energy / (total_energy + 1e-10)  #to avoid div by 0
        psd_norm = np.abs(coeff) / (np.sum(np.abs(coeff)) + 1e-10)
        ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))  # Shannon entropy

        # Compute per-coefficient statistics
        wavelet_features.update({
            f'{band_name}_energy': energy,
            f'{band_name}_rel_energy': rel_energy,
            f'{band_name}_entropy': ent,
            f'{band_name}_mean': np.mean(coeff),
            f'{band_name}_std': np.std(coeff),
            f'{band_name}_skew': skew(coeff),
            f'{band_name}_kurt': kurtosis(coeff),
        })

    return wavelet_features 

    
def extract_features(data, channel_info, config):
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
        return extract_multi_channel_features(data, channel_info, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return extract_single_channel_features(data, channel_info, config)


def extract_multi_channel_features(multi_channel_data, channel_info, config):
    """
    Extract features from multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.

    Students should expand this significantly!
    """
    eeg_fs = channel_info['eeg_fs']
    eog_fs = channel_info['eog_fs']
    emg_fs = channel_info['emg_fs']

    n_epochs = multi_channel_data['eeg'].shape[0]
    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = []

        # EEG features (2 channels)
        for ch in range(multi_channel_data['eeg'].shape[1]):
            eeg_signal = multi_channel_data['eeg'][epoch_idx, ch, :]
            eeg_features = extract_time_domain_features(eeg_signal)
            epoch_features.extend(list(eeg_features.values()))

            eeg_features= extract_frequency_domain_features(eeg_signal, eeg_fs, AR_method, Welch_method, wavelet_method)
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
            epoch_features.extend(list[Any](emg_features.values()))

        all_features.append(epoch_features)

    features = np.array(all_features)

    if config.CURRENT_ITERATION == 1:
        expected = 2 * 16  # 2 EEG channels × 3 features each
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
        for epoch in data:
            features = extract_time_domain_features(epoch)
            all_features.append(list(features.values()))
        features = np.array(all_features)

        print(f"{features.shape[1]} features extracted")
        #print(f"WARNING: Only {features.shape[1]} features extracted, target is 16 for iteration 1")
        #print("Students must implement the remaining time-domain features!")

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
