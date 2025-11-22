import config
from src.data_loader import load_all_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import train_classifier
from src.visualization import visualize_results
from src.visualization import plot_hypnogram
from src.visualization import plot_sample_epoch
from src.visualization import visualize_fft
from src.visualization import visualize_signal
from src.visualization import plot_confusion_matrix
from src.report import generate_report
from src.utils import save_cache, load_cache, add_contextual_features
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
import matplotlib.pyplot as plt 
import os
import sys
import io
import numpy as np
from sklearn.preprocessing import StandardScaler


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
    
    print(f"Multi-channel data loaded:")
    print(f"  EEG: {multi_channel_data['eeg'].shape}")
    print(f"  EOG: {multi_channel_data['eog'].shape}")
    print(f"  EMG: {multi_channel_data['emg'].shape}")
    print(f"Labels shape: {labels.shape}")
    ''' 
    example_edf_file = os.path.join(config.TRAINING_DIR, "R1.edf")
    example_xml_file = os.path.join(config.TRAINING_DIR, "R1.xml")
    if os.path.exists(example_xml_file):
        plot_hypnogram(example_xml_file)
    if os.path.exists(example_edf_file):
        plot_sample_epoch(example_edf_file, epoch_idx=0)
    
    plt.show()'''
   
    
    # 2. Preprocessing
    print("\n=== STEP 2: PREPROCESSING ===")
    preprocessed_data = None
    cache_filename_preprocess = f"preprocessed_data_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
        if preprocessed_data is not None:
            print("Loaded preprocessed data from cache")

    if preprocessed_data is None:
        preprocessed_data = preprocess(multi_channel_data, channel_info, config)
        print(f"Preprocessed EEG data shape: {preprocessed_data['eeg'].shape}")
        print(f"Preprocessed EOG data shape: {preprocessed_data['eog'].shape}")
        if config.USE_CACHE:
            save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
            print("Saved preprocessed data to cache")
    
    raw_signal = multi_channel_data['eog'][0,0,:]
    ''' 
    visualize_signal(raw_signal if isinstance(raw_signal, np.ndarray) else raw_signal, 
                 fs=channel_info['eog_fs'], title="Raw EOG Signal (Time Domain)")
    visualize_fft(raw_signal if isinstance(raw_signal, np.ndarray) else raw_signal, 
              fs=channel_info['eog_fs'], title="Raw EOG Signal FFT")
    visualize_signal(preprocessed_data[0] if isinstance(preprocessed_data, np.ndarray) 
                 else preprocessed_data['eog'][0,0,:], fs=channel_info['eog_fs'], title="Filtered EOG Signal (Time Domain)")
    visualize_fft(preprocessed_data[0] if isinstance(preprocessed_data, np.ndarray) 
              else preprocessed_data['eog'][0,0,:], fs=channel_info['eog_fs'], title="Filtered EOG Signal FFT")

    plt.show()'''
   

    # 3. Feature Extraction
    print("\n=== STEP 3: FEATURE EXTRACTION ===")
    features = None
    cache_filename_features = f"features_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        features = load_cache(cache_filename_features, config.CACHE_DIR)
        if features is not None:
            print("Loaded features from cache")

    if features is None:
        features = extract_features(preprocessed_data, channel_info ,config) 
        print(f"Extracted features shape: {features.shape}")
        if features.shape[1] == 0:
            print("⚠️  WARNING: No features extracted! Students must implement feature extraction.")
        if config.USE_CACHE:
            save_cache(features, cache_filename_features, config.CACHE_DIR)
            print("Saved features to cache")

    # 4. Feature Selection
    print("\n=== STEP 4: FEATURE SELECTION ===")
    selected_features = select_features(features, labels, config)
    #selected_features = features
    print(f"Selected features shape: {selected_features.shape}")

    print("Adding contextual features to SELECTED features...")
    selected_features = add_contextual_features(selected_features, n_prev=2, n_next=2)
    print(f"Final feature shape for training: {selected_features.shape}")

    # 5. Classification
    print("\n=== STEP 5: CLASSIFICATION ===")
    if selected_features.shape[1] > 0:
        scaler = StandardScaler()
        model, y_true_all, y_pred_all = train_classifier(selected_features, labels, all_record_ids, config, scaler=scaler) #modify by Sherry for classification(Strategy B)
        print(f"Trained {config.CLASSIFIER_TYPE} classifier")
        
        # Save model and scaler for inference
        model_filename = f"model_iter{config.CURRENT_ITERATION}.joblib"
        save_cache(model, model_filename, config.CACHE_DIR)
        print(f"Saved model to {model_filename}")
        
        scaler_filename = f"scaler_iter{config.CURRENT_ITERATION}.joblib"
        save_cache(scaler, scaler_filename, config.CACHE_DIR)
        print(f"Saved scaler to {scaler_filename}")
    else:
        print("⚠️  WARNING: Cannot train classifier - no features available!")
        print("Students must implement feature extraction first.")
        model = None

    # 6. Visualization
    print("\n=== STEP 6: VISUALIZATION ===")
    if model is not None:
        #plot_confusion_matrix(y_test, y_pred, class_names)
        visualize_results(y_true_all, y_pred_all, config)
        plt.show()
    else:
        print("Skipping visualization - no trained model")

    # 7. Report Generation
    print("\n=== STEP 7: PROCESSING LOG & REPORT GENERATION ===")

    # Restore the original stdout
    sys.stdout = original_stdout

    # Get the captured output from the buffer
    processing_log = stdout_buffer.getvalue()   
     
    if model is not None:
        generate_report(model, selected_features, labels, config, processing_log, y_true_all, y_pred_all)
    else:
        print("Skipping report - no trained model")

    print("\n" + "="*50)
    print("PIPELINE FINISHED")
    if model is None:
        print("⚠️  Students need to implement missing components!")
    print("="*50)

if __name__ == "__main__":
    main()