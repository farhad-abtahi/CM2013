from scipy.signal import butter, lfilter, iirnotch, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import welch

import numpy as np

def bandpass_filter(data, low_cutoff, high_cutoff, fs):
    """
    EXAMPLE IMPLEMENTATION: Simple low-pass Butterworth filter.

    Students should understand this basic filter and consider:
    - Is 40Hz the right cutoff for EEG? https://pmc.ncbi.nlm.nih.gov/articles/PMC10312706/ 
         Most EEG signals of interest (delta, theta, alpha, beta) are below 40 Hz. Gamma activity is above 30-40 Hz. https://www.sciencedirect.com/science/article/pii/S2666720724001152
    - What about high-pass filtering? https://pressrelease.brainproducts.com/eeg-artifacts-handling-in-analyzer/ 
        High-pass filter attenuates frequencies below the low-cutoff. Common cutoffs in EEG are between 0.1 or 0.5 Hz to reduce drifts such as body sway, or skin potentials. 
    Likewise, they dramatically reduce offsets as very slow oscillations in the data.
    - Should you use bandpass instead? 
        Yes, that way we combine high-pass and low-pass to keep only a specific frequency range (e.g., 0.5 - 40 Hz).
    - What about notch filtering for powerline interference?  
        The nocth filter aim is to attenuate line noise at 50 or 60 Hz. Removes electrical interference, often from power lines or electrical equipment.
        As we have used a band-pass filter which our upper cutoff is below 50/60 Hz ( 40 Hz), we are already removing most or all of the powerline noise. In this case, a notch filter at 50/60 Hz is usually not necessary, 
    Args:
        data (np.ndarray): The input signal.
        cutoff (float): The cutoff frequency of the filter.
        fs (int): The sampling frequency of the signal.
        order (int): The order of the filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    # TODO: Students may want to implement additional filtering:
    # - High-pass filter to remove DC drift
    # - Notch filter for 50/60 Hz powerline noise
    # - Bandpass filter (e.g., 0.5-40 Hz for EEG)

    nyquist = 0.5 * fs
    #To test the Zero padding with different orders and padtypes
    orders = [2,4,6]
    padtypes = ['odd', 'even', 'constant', None]
    plot_idx = 1

    plt.figure(figsize=(14, 10), constrained_layout=True)
    for order in orders:
        b, a = butter(order, [low_cutoff/nyquist, high_cutoff/nyquist], btype='band', analog=False)
        padlen = 3 * order
        for padtype in padtypes:
            y = filtfilt(b, a, data, padtype=padtype, padlen=padlen)
            plt.subplot(len(orders), len(padtypes), plot_idx)
            plt.plot(y[:fs*5])
            plt.title(f'order={order}, padtype={padtype}')
            plot_idx += 1
    plt.suptitle('Effect of Filter Order and Padding Type on Filtered Signal Edges')
    plt.show()
    return y

def notch_filter(data, notch_freq, eeg_fs):
    quality_factor = 30  
    print(f"Applying notch filter at {notch_freq} Hz with eeg_fs={eeg_fs}")
    b, a = iirnotch(notch_freq, quality_factor, eeg_fs)
    filtered_signal = filtfilt(b, a, data)
    return filtered_signal


def preprocess(data, config, channel_info):
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
        return preprocess_multi_channel(data, config, channel_info)
    else:
        print("Processing single-channel data (backward compatibility)")
        return preprocess_single_channel(data, config)


def preprocess_multi_channel(multi_channel_data, config, channel_info):
    """
    Preprocess multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.
    Each channel type may have different sampling rates and require different processing.
    """
    preprocessed_data = {}

    # Process EEG channels (2 channels)
    print("preprocess_multo_channel function")
    eeg_data = multi_channel_data['eeg']
    eeg_fs = channel_info['eeg_fs']  # Actual sampling rate: 125 Hz (TODO: Get from channel_info)
    preprocessed_eeg = np.zeros_like(eeg_data)

    for ch in range(eeg_data.shape[1]): #eeg_data: n_epochs, n_channels, n_samples_per_epoch
        for epoch in range(eeg_data.shape[0]):
            print(f"Processing channel {ch}, epoch {epoch}")
            signal = eeg_data[epoch, ch, :] #eg: singal = eeg_data[0,1,:] - all samples from channel 1(C4-A1) in the first epoch
            # Apply EEG-specific preprocessing
            filtered_signal = bandpass_filter(signal, config.HIGH_PASS_FILTER_FREQ, config.LOW_PASS_FILTER_FREQ, eeg_fs)
            # TODO: Students should add bandpass filter, artifact removal. I have changed the low pass for a bandpass filter, and added a notch filter
            # 2. noise 50/60 Hz and armonics, notch filter
            for notch_freq in [50, 100, 150]:
                if notch_freq < eeg_fs / 2:
                    preprocessed_data = notch_filter(preprocessed_data, notch_freq, eeg_fs)
                else:
                    print(f"Skipping notch filter at {notch_freq} Hz because it exceeds Nyquist frequency")
                    
            # Run validation ONCE for the first channel and epoch to save time
            if ch == 0 and epoch == 0:
                print("Running validation for first EEG epoch...")
                validate_filtering(signal, filtered_signal, eeg_fs) 

            preprocessed_eeg[epoch, ch, :] = filtered_signal

    preprocessed_data['eeg'] = preprocessed_eeg


    if config.CURRENT_ITERATION >= 2:  # EOG starts in iteration 2
        # Process EOG channels (2 channels) - may need different filtering
        eog_data = multi_channel_data['eog']
        eog_fs = channel_info['eog_fs']   # Actual sampling rate: 50 Hz (TODO: Get from channel_info)
        preprocessed_eog = np.zeros_like(eog_data)

        for ch in range(eog_data.shape[1]):
            for epoch in range(eog_data.shape[0]):
                signal = eog_data[epoch, ch, :]
                # EOG may need different filter settings (preserve slow eye movements)
                filtered_signal = bandpass_filter(signal, 30, eog_fs)  # Lower cutoff for EOG
                preprocessed_eog[epoch, ch, :] = filtered_signal

        preprocessed_data['eog'] = preprocessed_eog

    if config.CURRENT_ITERATION >= 3:  # EMG starts in iteration 3
        # Process EMG channel (1 channel) - may need higher frequency preservation
        emg_data = multi_channel_data['emg']
        emg_fs = channel_info['emg_fs']   # Actual sampling rate: 125 Hz (TODO: Get from channel_info)
        preprocessed_emg = np.zeros_like(emg_data)

        for epoch in range(emg_data.shape[0]):
            signal = emg_data[epoch, 0, :]
            # EMG needs higher frequency content preserved (muscle activity)
            filtered_signal = bandpass_filter(signal, 70, emg_fs)  # Higher cutoff for EMG
            preprocessed_emg[epoch, 0, :] = filtered_signal

        preprocessed_data['emg'] = preprocessed_emg
        print("Multi-channel preprocessing applied to EEG + EOG + EMG")
    elif config.CURRENT_ITERATION >= 2:
        print("Iteration 2: Processing EEG + EOG channels")
    else:
        print("Iteration 1: Processing EEG channels only")

    # TODO: Students should add:
    # - Channel-specific artifact removal
    # - Cross-channel artifact detection
    # - Signal quality assessment
    # - Normalization per channel type

    return preprocessed_data


def preprocess_single_channel(data, config):
    """
    Backward compatibility for single-channel preprocessing.
    """
    if config.CURRENT_ITERATION == 1:
        # EXAMPLE: Very basic low-pass filter (students should expand)
        fs = 125  # Actual EEG sampling rate: 125 Hz (TODO: Get from data/config)
        preprocessed_data = bandpass_filter(data, config.HIGH_PASS_FILTER_FREQ, config.LOW_PASS_FILTER_FREQ, fs)
        # 2. noise 50/60 Hz and armonics, notch filter
        for notch_freq in [50, 100, 150]:
            if notch_freq < fs / 2:
                preprocessed_data = notch_filter(preprocessed_data, notch_freq, fs)
            else:
                print(f"Skipping notch filter at {notch_freq} Hz because it exceeds Nyquist frequency")

        # Run validation ONCE for the first channel and epoch to save time
        print("Running validation for first EEG epoch...")
        validate_filtering(data, preprocessed_data, fs) 

    elif config.CURRENT_ITERATION == 2:
        print("TODO: Implement enhanced preprocessing for iteration 2")
        preprocessed_data = data  # Placeholder

    elif config.CURRENT_ITERATION >= 3:
        print("TODO: Students should use multi-channel data format for iteration 3+")
        preprocessed_data = data  # Placeholder

    else:
        raise ValueError(f"Invalid iteration: {config.CURRENT_ITERATION}")

    return preprocessed_data


def validate_filtering(original, filtered, fs):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    # Reduce to 1D signal (select first epoch if shape is (epochs, samples))
    if original.ndim > 1:
        original = original[0]
    if filtered.ndim > 1:
        filtered = filtered[0]

    # Print summary statistics
    print("Original stats: mean={:.4e}, min={:.4e}, max={:.4e}".format(np.mean(original), np.min(original), np.max(original)))
    print("Filtered stats: mean={:.4e}, min={:.4e}, max={:.4e}".format(np.mean(filtered), np.min(filtered), np.max(filtered)))

    # 1. Check baseline drift removal (mean ~ 0)
    mean_before = np.mean(original)
    mean_after = np.mean(filtered)
    print(f"Mean before: {mean_before:.4f}, Mean after: {mean_after:.4f}")

    # 2. Check correlation (ensures no major phase shift)
    try:
        corr = np.corrcoef(original, filtered)[0, 1]
    except Exception:
        corr = float('nan')
    print(f"Correlation between original and filtered: {corr:.4f}")

    # 3. Time-domain overlay plot
    plt.figure(figsize=(12,5))
    plt.plot(original, label='Original')
    plt.plot(filtered, label='Filtered')
    plt.title('Time-domain Signal: Original vs Filtered')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

    # 4. Power spectrum comparison
    f_orig, Pxx_orig = welch(original, fs)
    f_filt, Pxx_filt = welch(filtered, fs)
    plt.figure(figsize=(10, 5))
    plt.semilogy(f_orig, Pxx_orig, label="Original")
    plt.semilogy(f_filt, Pxx_filt, label="Filtered")
    plt.title("PSD Comparison (Full Band)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.show()

    # 5. Zoom on delta band (0-5 Hz)
    plt.figure(figsize=(8, 4))
    plt.semilogy(f_orig, Pxx_orig, label='Original')
    plt.semilogy(f_filt, Pxx_filt, label='Filtered')
    plt.xlim(0, 5)
    plt.title('PSD: Delta Band Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.show()

    # 6. Zoom on 50 Hz region
    plt.figure(figsize=(8, 4))
    plt.semilogy(f_orig, Pxx_orig, label='Original')
    plt.semilogy(f_filt, Pxx_filt, label='Filtered')
    plt.xlim(45, 55)
    plt.title('PSD: 50 Hz Region Comparison')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.legend()
    plt.show()

    # 7. Compute and check powerline attenuation safely
    notch_band = (f_orig > 45) & (f_orig < 55)
    power_orig = np.sum(Pxx_orig[notch_band])
    power_filt = np.sum(Pxx_filt[notch_band])
    eps = 1e-12
    if power_orig < eps:
        print("WARNING: No original power at 50 Hz band; skipping ratio calculation.")
        power_ratio_50Hz = 0
    else:
        power_ratio_50Hz = power_filt / power_orig
    print(f"Powerline noise (50Hz band) reduced to {power_ratio_50Hz * 100:.2f}% of original")

    # 8. Final pass/fail checks
    if abs(mean_after) < 0.05 and (abs(corr) > 0.8) and (power_ratio_50Hz < 0.2):
        print("Validation PASSED ✅")
    else:
        print("Validation WARN ⚠️ Some artifacts remain.")

lowpass_filter = bandpass_filter