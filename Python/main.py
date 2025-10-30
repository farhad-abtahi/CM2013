import config
import numpy as np
import pandas as pd
from src.data_loader import load_all_training_data, load_holdout_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import train_classifier
from src.visualization import visualize_results
from src.visualization import plot_hypnogram
from src.visualization import plot_sample_epoch
from src.visualization import visualize_fft
from src.visualization import visualize_signal
from src.report import generate_report
from src.utils import save_cache, load_cache
import os
import sys
import io
from glob import glob


def main():
    # Create a string buffer
    stdout_buffer = io.StringIO()

    # Save the original stdout
    original_stdout = sys.stdout

    # Redirect stdout to the buffer
    sys.stdout = stdout_buffer 

    print("\n=== PROCESSING LOG ===")

    print(f"--- Sleep Scoring Pipeline - Iteration {config.CURRENT_ITERATION} ---")

    # 1. Load Data
    # Example uses R1.edf and R1.xml from training directory
    print("\n=== STEP 1: DATA LOADING ===")

    #Loading all data
    multi_channel_data, labels, all_record_ids, channel_info = load_all_training_data(config.TRAINING_DIR)
    print(f"Total data loaded: {labels.shape[0]} epochs from {len(np.unique(all_record_ids))} recordings")
    print(f"Multi-channel data structure:")
    if 'eeg' in multi_channel_data:
        print(f"  EEG: {multi_channel_data['eeg'].shape}")
    if 'eog' in multi_channel_data:
        print(f"  EOG: {multi_channel_data['eog'].shape}")
    if 'emg' in multi_channel_data:
        print(f"  EMG: {multi_channel_data['emg'].shape}")

    print("Plotting hypnogram and sample epoch for R1 (example)...")
    example_edf_file = os.path.join(config.TRAINING_DIR, "R1.edf")
    example_xml_file = os.path.join(config.TRAINING_DIR, "R1.xml")
    if os.path.exists(example_xml_file):
        plot_hypnogram(example_xml_file)
    if os.path.exists(example_edf_file):
        plot_sample_epoch(example_edf_file, epoch_idx=0)

    # 2. Preprocessing
    print("\n=== STEP 2: PREPROCESSING ===")
    preprocessed_data = None
    cache_filename_preprocess = f"preprocessed_data_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
        if preprocessed_data is not None:
            print("Loaded preprocessed data from cache")

    if preprocessed_data is None:
        preprocessed_data = preprocess(multi_channel_data, config)
        print(f"Preprocessed data shape: {preprocessed_data['eeg'].shape}")
        if config.USE_CACHE:
            save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
            print("Saved preprocessed data to cache")
    
    fs=channel_info.get('eeg_fs',125)
    raw_signal = multi_channel_data['eeg'][0,0,:]
    visualize_signal(raw_signal, fs=fs, title="Raw EEG Signal (Ch 0, Epoch 0)")
    visualize_fft(raw_signal, fs=fs, title="Raw EEG Signal FFT (Ch 0, Epoch 0)")
    filtered_signal = preprocessed_data['eeg'][0, 0, :]
    visualize_signal(filtered_signal, fs=fs, title="Filtered EEG Signal (Ch 0, Epoch 0)")
    visualize_fft(filtered_signal, fs=fs, title="Filtered EEG Signal FFT (Ch 0, Epoch 0)")

    # 3. Feature Extraction
    print("\n=== STEP 3: FEATURE EXTRACTION ===")
    features = None
    cache_filename_features = f"features_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        features = load_cache(cache_filename_features, config.CACHE_DIR)
        if features is not None:
            print("Loaded features from cache")

    if features is None:
        features = extract_features(preprocessed_data, config)
        print(f"Extracted features shape: {features.shape}")
        if features.shape[1] == 0:
            print("⚠️  WARNING: No features extracted! Students must implement feature extraction.")
        if config.USE_CACHE:
            save_cache(features, cache_filename_features, config.CACHE_DIR)
            print("Saved features to cache")

    # 4. Feature Selection
    print("\n=== STEP 4: FEATURE SELECTION ===")
    selected_features=features #For iteration 1 we don't need to select data
    #selected_features = select_features(features, labels, config)
    print(f"Selected features shape: {selected_features.shape}")

    # 5. Classification
    print("\n=== STEP 5: CLASSIFICATION ===")
    if selected_features.shape[1] > 0:
        model = train_classifier(selected_features, labels, all_record_ids, config) #modify by Sherry for classification(Strategy B)
        print(f"Trained {config.CLASSIFIER_TYPE} classifier")
    else:
        print("⚠️  WARNING: Cannot train classifier - no features available!")
        #print("Students must implement feature extraction first.")
        model = None

    # 6. Visualization
    print("\n=== STEP 6: VISUALIZATION ===")
    if model is not None:
        visualize_results(model, selected_features, labels, config)
    else:
        print("Skipping visualization - no trained model")

    # 7. Report Generation
    print("\n=== STEP 7: PROCESSING LOG & REPORT GENERATION ===")

    #print("\n" + "="*50)
    #print("PIPELINE FINISHED")
    #if model is None:
    #    print("⚠️  Students need to implement missing components!")
    #print("="*50)

    # 8. Holdout Prediction(add by Sherry)
    print("\n=== STEP 8: HOLDOUT PREDICTION ===")
    if model is not None:
        holdout_files = sorted(glob(os.path.join(config.HOLDOUT_DIR, '*.edf')))
        if not holdout_files:
            print(f"No holdout EDF files found in {config.HOLDOUT_DIR}")
        else:
            print(f"Found {len(holdout_files)} holdout files for prediction.")
            all_submission_data = []
            for edf_file in holdout_files:
                try:
                    print(f"\nProcessing holdout file: {os.path.basename(edf_file)}...")
                    
                    holdout_data, record_info = load_holdout_data(edf_file)
                    record_id = record_info['record_id']
                    holdout_preprocessed = preprocess(holdout_data, config)
                    holdout_features = extract_features(holdout_preprocessed, config)
                    
                    print(f"  Predicting {holdout_features.shape[0]} epochs for {record_id}...")
                    predictions = model.predict(holdout_features)
                    for epoch_index, prediction in enumerate(predictions):
                        all_submission_data.append({
                            'record_id': record_id,
                            'epoch': epoch_index,
                            'prediction': prediction
                            })
                except Exception as e:
                    print(f"  ERROR processing {os.path.basename(edf_file)}: {e}")
            
            if all_submission_data:
                submission_df = pd.DataFrame(all_submission_data)
                submission_file_path = config.SUBMISSION_FILE 
                submission_df.to_csv(submission_file_path, index=False)
                print(f"Submission file created successfully at: {submission_file_path}")
                print(f"Total predictions: {len(submission_df)}")
            else:
                print("No predictions were generated.")

    else:
        print("Skipping holdout prediction - no trained model")


    
    # Restore the original stdout
    sys.stdout = original_stdout

    # Get the captured output from the buffer
    processing_log = stdout_buffer.getvalue()

    if model is not None:
        generate_report(model, selected_features, labels, config, processing_log)
    else:
        print("Skipping report - no trained model")

    print("\n" + "="*50)
    print("PIPELINE FINISHED")  

if __name__ == "__main__":
    main()