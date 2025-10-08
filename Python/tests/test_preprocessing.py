import numpy as np
from src.preprocessing import lowpass_filter, preprocess
import config

def test_lowpass_filter():
    fs = 100
    cutoff = 10
    data = np.sin(2 * np.pi * 5 * np.arange(0, 10, 1/fs)) + np.sin(2 * np.pi * 20 * np.arange(0, 10, 1/fs))
    filtered_data = lowpass_filter(data, cutoff, fs)
    assert isinstance(filtered_data, np.ndarray)
    assert filtered_data.shape == data.shape
    # Basic check: ensure some attenuation of high frequency component
    # This is a very simple check and might need more sophisticated validation
    assert np.std(filtered_data) < np.std(data) # Expect some reduction in signal power

def test_preprocess_iteration_1():
    # Dummy data: 20 epochs, 3000 samples per epoch
    eeg_data = np.random.randn(20, 3000)
    config.CURRENT_ITERATION = 1
    preprocessed_data = preprocess(eeg_data, config)
    assert isinstance(preprocessed_data, np.ndarray)
    assert preprocessed_data.shape == eeg_data.shape

def test_preprocess_other_iteration():
    # Dummy data: 20 epochs, 3000 samples per epoch
    eeg_data = np.random.randn(20, 3000)
    config.CURRENT_ITERATION = 99 # An iteration not explicitly handled
    preprocessed_data = preprocess(eeg_data, config)
    assert isinstance(preprocessed_data, np.ndarray)
    assert preprocessed_data.shape == eeg_data.shape
    # For other iterations, it should return raw data, so it should be very similar
    assert np.allclose(preprocessed_data, eeg_data) # Check if it's essentially the same
