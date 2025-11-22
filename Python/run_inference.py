import config
from src.data_loader import load_holdout_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.inference import make_inference, generate_submission_file
from src.utils import save_cache, load_cache, add_contextual_features
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest
import pandas as pd
import os
import joblib
import glob
import re
import numpy as np

def extract_record_number(filename):
    """
    Extract record number from filename (e.g., 'H1.edf' -> 1, 'H09.edf' -> 9, 'H10.edf' -> 10)
    """
    # Extract the number after 'H' prefix
    match = re.search(r'H(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def convert_record_info_to_channel_info(record_info):
    """
    Convert record_info from load_holdout_data to channel_info format for preprocessing.
    """
    channel_info = {
        'epoch_length': record_info['epoch_length']
    }
    
    # Convert sampling_rates dict to individual keys
    if 'sampling_rates' in record_info:
        if 'eeg' in record_info['sampling_rates']:
            channel_info['eeg_fs'] = record_info['sampling_rates']['eeg']
        if 'eog' in record_info['sampling_rates']:
            channel_info['eog_fs'] = record_info['sampling_rates']['eog']
        if 'emg' in record_info['sampling_rates']:
            channel_info['emg_fs'] = record_info['sampling_rates']['emg']
    
    return channel_info
def apply_feature_selection(features, config):
    print(f"Applying saved feature selection for iteration {config.CURRENT_ITERATION}...")
    print(f"Input features shape: {features.shape}")

    # =============================
    # 1. Load variance mask
    # =============================
    varmask = load_cache(f"varmask_iter{config.CURRENT_ITERATION}.joblib", config.CACHE_DIR)
    if varmask is None:
        raise ValueError("ERROR: varmask_iterX.joblib not found!")
    features = features[:, varmask]
    print("After variance mask:", features.shape)

    # =============================
    # 2. Load correlation mask
    # =============================
    corrmask = load_cache(f"corrmask_iter{config.CURRENT_ITERATION}.joblib", config.CACHE_DIR)
    if corrmask is None:
        raise ValueError("ERROR: corrmask_iterX.joblib not found!")
    features = features[:, corrmask]
    print("After correlation mask:", features.shape)

    # =============================
    # 3. Apply saved SelectKBest
    # =============================
    selector = load_cache(f"feature_selector_iter{config.CURRENT_ITERATION}.joblib", config.CACHE_DIR)
    if selector is None:
        raise ValueError("ERROR: feature selector not found!")

    features_selected = selector.transform(features)
    print("After KBest:", features_selected.shape)

    return features_selected

def run_inference():
    print(f"--- Sleep Scoring Inference - Iteration {config.CURRENT_ITERATION} ---")

    # Load the trained model (assuming it was saved during training)
    model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
    model = load_cache(model_filename, config.CACHE_DIR)
    if model is None:
        print("Error: Trained model not found. Please run main.py first to train a model.")
        return

    # 1. Load Hold-out Data
    # Iterate through all .edf files in the holdout directory
    holdout_edf_files = glob.glob(os.path.join(config.HOLDOUT_DIR, "*.edf"))
    holdout_edf_files = [f for f in holdout_edf_files if os.path.isfile(f)]  # Exclude directories
    
    if not holdout_edf_files:
        print(f"Error: No EDF files found in {config.HOLDOUT_DIR}")
        return
    
    # Sort files to ensure consistent processing order
    holdout_edf_files.sort()
    print(f"Found {len(holdout_edf_files)} holdout EDF files to process")
    
    # Collect all predictions, record numbers, and epoch numbers
    all_predictions = []
    all_record_numbers = []
    all_epoch_numbers = []
    
    # Process each holdout file
    for holdout_edf_file in holdout_edf_files:
        filename = os.path.basename(holdout_edf_file)
        record_number = extract_record_number(filename)
        
        if record_number is None:
            print(f"Warning: Could not extract record number from {filename}, skipping...")
            continue
        
        print(f"\nProcessing {filename} (Record {record_number})...")
        
        # Load holdout data
        holdout_eeg_data, record_info = load_holdout_data(holdout_edf_file)
        
        # Convert record_info to channel_info format
        channel_info = convert_record_info_to_channel_info(record_info)
        
        # 2. Preprocessing (using the same logic as training)
        preprocessed_holdout_data = None
        cache_filename_preprocess_holdout = f"preprocessed_holdout_{record_number}_iter{config.CURRENT_ITERATION}.joblib"
        if config.USE_CACHE:
            preprocessed_holdout_data = load_cache(cache_filename_preprocess_holdout, config.CACHE_DIR)
        
        if preprocessed_holdout_data is None:
            preprocessed_holdout_data = preprocess(holdout_eeg_data, channel_info, config)
            if config.USE_CACHE:
                save_cache(preprocessed_holdout_data, cache_filename_preprocess_holdout, config.CACHE_DIR)

        # 3. Feature Extraction (using the same logic as training)
        holdout_features = None
        cache_filename_features_holdout = f"features_holdout_{record_number}_iter{config.CURRENT_ITERATION}.joblib"
        if config.USE_CACHE:
            holdout_features = load_cache(cache_filename_features_holdout, config.CACHE_DIR)

        if holdout_features is None:
            holdout_features = extract_features(preprocessed_holdout_data, channel_info, config)
            if config.USE_CACHE:
                save_cache(holdout_features, cache_filename_features_holdout, config.CACHE_DIR)
        
        print(f"Extracted features shape: {holdout_features.shape}")

        # 4. Feature Selection (using the same logic as training)
        selected_features = apply_feature_selection(holdout_features, config)
        
        # 5. Add Contextual Features (using the same logic as training)
        print("Adding contextual features...")
        selected_features = add_contextual_features(selected_features, n_prev=2, n_next=2)
        print(f"Final feature shape after context: {selected_features.shape}")
        
        # 6. Scale features (using saved scaler from training)
        scaler_filename = f"scaler_iter{config.CURRENT_ITERATION}.joblib"
        scaler = load_cache(scaler_filename, config.CACHE_DIR)
        if scaler is None:
            print("Warning: Scaler not found. Using StandardScaler without saved parameters.")
            scaler = StandardScaler()
            # This shouldn't happen if training was done correctly
            print("Error: Scaler should be saved during training. Please run main.py first.")
            return
        
        selected_features_scaled = scaler.transform(selected_features)

        # 7. Make Inference
        predictions = make_inference(model, selected_features_scaled, config)
        
        # Collect predictions with corresponding record and epoch numbers
        n_epochs = len(predictions)
        all_predictions.extend(predictions)
        all_record_numbers.extend([record_number] * n_epochs)
        all_epoch_numbers.extend(list(range(n_epochs)))
        
        print(f"Generated {n_epochs} predictions for record {record_number}")

    # 5. Generate Submission File
    if all_predictions:
        all_predictions = np.array(all_predictions)
        generate_submission_file(all_predictions, all_record_numbers, all_epoch_numbers, config)
        print(f"\nTotal: {len(all_predictions)} predictions from {len(holdout_edf_files)} files")
    else:
        print("Error: No predictions generated!")

    print("--- Inference Finished ---")

if __name__ == "__main__":
    run_inference()
