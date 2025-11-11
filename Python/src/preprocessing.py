from scipy.signal import butter, lfilter, filtfilt, iirnotch, welch
import numpy as np

def lowpass_filter(data, cutoff, fs, order=5):
    """
    EXAMPLE IMPLEMENTATION: Simple low-pass Butterworth filter.

    Students should understand this basic filter and consider:
    - Is 40Hz the right cutoff for EEG?
    - What about high-pass filtering?
    - Should you use bandpass instead?
    - What about notch filtering for powerline interference?

    Args:
        data (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    # DONE: Students may want to implement additional filtering:
    # - High-pass filter to remove DC drift
    # - Notch filter for 50/60 Hz powerline noise
    # - Bandpass filter (e.g., 0.5-40 Hz for EEG)

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def highpass_filter(data, cutoff, fs, order=5): #Highpass filter add by Sherry
    nyquist = 0.5 *fs
    normal_cutoff = cutoff/nyquist
    b,a =butter(order, normal_cutoff, btype="high", analog=False)
    y = filtfilt(b,a,data)
    return y

def bandpass_filter(data, lowcut , highcut, fs, order):# Bandpass filter add by Shuxuan
    nyquist = 0.5 * fs
    normal_lowcut = lowcut / nyquist
    normal_highcut = highcut/nyquist
    b, a = butter(order, [normal_lowcut, normal_highcut], btype='band', analog=False)
    y = filtfilt(b, a, data)
    return y

def notch_filter(data, to_be_removed, fs, q_factor, no_harmonics): # Notch filter added by Zuzanna
    nyquist_criterion = fs/2.0
    filtered = data.copy()  #start from original signal

    #going through base freq and its harmonics
    for h in range(1, no_harmonics + 1):
        notch_freq = h * to_be_removed

        if notch_freq >= nyquist_criterion:
            #print(f"{notch_freq} skipped")
            continue

        b, a = iirnotch(w0=notch_freq, Q=q_factor, fs=fs)
        filtered = lfilter(b, a, filtered)

    return filtered

def compute_welch_psd(signal, fs, nperseg=None, noverlap=None, window='hann', nfft=None, scaling='density'):
    if nperseg is None:
        nperseg = int(4 * fs)
    
    if noverlap is None:
        noverlap = int(0.5 * nperseg)

    freqs, psd = welch(
        signal, 
        fs=fs, 
        window=window, 
        nperseg=nperseg, 
        noverlap=noverlap, 
        nfft=nfft, 
        scaling=scaling
    )
    
    return freqs, psd

def preprocess(data, channel_info, config):
    """
    STUDENT IMPLEMENTATION AREA: Preprocess data based on current iteration.

    This function should handle both single-channel and multi-channel data
    (2 EEG + 2 EOG + 1 EMG channels) based on the data structure.

    Args:
        data: Either np.ndarray (single-channel) or dict (multi-channel)
        config (module): The configuration module.

    Returns:
        Same format as input: preprocessed data.
    """
    print(f"Preprocessing data for iteration {config.CURRENT_ITERATION}...")

    # Detect data format
    is_multi_channel = isinstance(data, dict) and 'eeg' in data

    if is_multi_channel:
        print("Processing multi-channel data (EEG + EOG + EMG)")
        return preprocess_multi_channel(data, channel_info, config)
    else:
        print("Processing single-channel data (backward compatibility)")
        return preprocess_single_channel(data, channel_info, config)


def preprocess_multi_channel(multi_channel_data, channel_info, config):
    """
    Preprocess multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.
    Each channel type may have different sampling rates and require different processing.
    """
    preprocessed_data = {}

    # Process EEG channels (2 channels)
    eeg_data = multi_channel_data['eeg']
    eeg_fs = channel_info['eeg_fs']  # Actual sampling rate: 125 Hz (DONE: Get from channel_info)
    to_be_removed = 50 #for notch filter; can be 60
    q_factor = 30 #for notch filter; the higher its value, the narrower notch; typical range for EEG: 30-50
    no_harmonics = 2 #for notch filter
    preprocessed_eeg = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[1]):
        for epoch in range(eeg_data.shape[0]):
            signal = eeg_data[epoch, ch, :]
            # Apply EEG-specific preprocessing
            filtered_signal = lowpass_filter(signal, config.LOW_PASS_FILTER_EEG_FREQ, eeg_fs)
            filtered_signal = highpass_filter(filtered_signal, config.HIGH_PASS_FILTER_FREQ, eeg_fs) #Highpass filter add by Sherry
            filtered_signal = notch_filter(filtered_signal, to_be_removed, eeg_fs, q_factor, no_harmonics) #new
            filtered_signal = bandpass_filter(filtered_signal, config.HIGH_PASS_FILTER_FREQ, config.LOW_PASS_FILTER_EEG_FREQ, eeg_fs, order = 4)# Bandpass filter add by Shuxuan
            # DONE: Students should add bandpass filter, artifact removal
            preprocessed_eeg[epoch, ch, :] = filtered_signal

    preprocessed_data['eeg'] = preprocessed_eeg

    if config.CURRENT_ITERATION >= 2:  # EOG starts in iteration 2
        # Process EOG channels (2 channels) - may need different filtering
        eog_data = multi_channel_data['eog']
        eog_fs = channel_info['eog_fs']  # Actual sampling rate: 50 Hz (DONE: Get from channel_info)
        preprocessed_eog = np.zeros_like(eog_data)

        for ch in range(eog_data.shape[1]):
            for epoch in range(eog_data.shape[0]):
                signal = eog_data[epoch, ch, :]
                # EOG may need different filter settings (preserve slow eye movements)
                filtered_signal = lowpass_filter(signal, 30, eog_fs)  # Lower cutoff for EOG
                filtered_signal = notch_filter(filtered_signal, to_be_removed, eog_fs, q_factor, no_harmonics) #new
                filtered_signal = bandpass_filter(filtered_signal, config.HIGH_PASS_FILTER_FREQ, config.LOW_PASS_FILTER_EOG_FREQ, eog_fs, order = 4)
                preprocessed_eog[epoch, ch, :] = filtered_signal
                
        preprocessed_data['eog'] = preprocessed_eog

    if config.CURRENT_ITERATION >= 3:  # EMG starts in iteration 
        # Process EMG channel (1 channel) - may need higher frequency preservation
        emg_data = multi_channel_data['emg']
        emg_fs = channel_info['emg_fs']  # Actual sampling rate: 125 Hz (DONE: Get from channel_info)
        preprocessed_emg = np.zeros_like(emg_data)

        for epoch in range(emg_data.shape[0]):
            signal = emg_data[epoch, 0, :]
            # EMG needs higher frequency content preserved (muscle activity)
            filtered_signal = lowpass_filter(signal, 70, emg_fs)  # Higher cutoff for EMG
            preprocessed_emg[epoch, 0, :] = filtered_signal

        preprocessed_data['emg'] = preprocessed_emg
        print("Multi-channel preprocessing applied to EEG + EOG + EMG")
    elif config.CURRENT_ITERATION >= 2:
        print("Iteration 2: Processing EEG + EOG channels")
    else:
        print("Iteration 1: Processing EEG channels only")

    # TODO: Students should add:
    # - Channel-specific artifact removal -> DONE
    # - Cross-channel artifact detection
    # - Signal quality assessment
    # - Normalization per channel type

    return preprocessed_data


def preprocess_single_channel(data, channel_info, config):
    """
    Backward compatibility for single-channel preprocessing.
    """
    if config.CURRENT_ITERATION == 1:
        # EXAMPLE: Very basic low-pass filter (students should expand)
        fs = channel_info['eeg_fs'] # Actual EEG sampling rate: 125 Hz (DONE: Get from data/config)
        #changed by Shuxuan
        #preprocessed_data = lowpass_filter(data, config.LOW_PASS_FILTER_FREQ, fs)
        #preprocessed_data = highpass_filter(data, config.HIGH_PASS_FILTER_FREQ, fs) #Highpass filter add by Sherry
        preprocessed_data = notch_filter(data, to_be_removed = 50, fs = 125, q_factor = 30, no_harmonics = 2)
        preprocessed_data = bandpass_filter(preprocessed_data, config.HIGH_PASS_FILTER_FREQ, config.LOW_PASS_FILTER_FREQ, fs = 125, order=4)
    elif config.CURRENT_ITERATION == 2:
        print("TODO: Implement enhanced preprocessing for iteration 2")
        fs=channel_info['eog_fs']
        preprocessed_data = data  # Placeholder

    elif config.CURRENT_ITERATION >= 3:
        print("TODO: Students should use multi-channel data format for iteration 3+")
        preprocessed_data = data  # Placeholder

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return preprocessed_data
