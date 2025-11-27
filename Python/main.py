# "C:\Users\prart\Signals\Scripts\python.exe"
import config
from src.data_loader import load_training_data
from src.preprocessing import preprocess
from src.feature_extraction import extract_features
from src.feature_selection import select_features
from src.classification import train_classifier
from src.visualization import visualize_results
from src.report import generate_report
from src.utils import save_cache, load_cache
from pathlib import Path
import os
import sys
import io
import numpy as np
import matplotlib.pyplot as plt

def main():
 

    print("\n=== PROCESSING LOG ===")

    print(f"--- Sleep Scoring Pipeline - Iteration {config.CURRENT_ITERATION} ---")

    # 1. Load Data
    # Example uses R1.edf and R1.xml - students should adapt for their dataset
    print("\n=== STEP 1: DATA LOADING ===")
    training_dir = config.TRAINING_DIR
    edf_file = list(Path(training_dir).glob('*.edf')) #this looks for edf files in data/training

    all_epochs = []
    all_labels = []
    all_record_ids = []
    #I have added this:
    multi_channel_list = [] 

    for edf_path in edf_file:
        xml_path = edf_path.with_suffix('.xml') #you get the xml file related to the edf file
        record_id = edf_path.stem #you extract the name of the file without the extension, R1.edf - R1
        
    #edf_file = os.path.join(config.SAMPLE_DIR, "S01.edf")  # Example EDF file
    #xml_file = os.path.join(config.SAMPLE_DIR, "S01.xml")  # Corresponding annotation file

    # Handle both new multi-channel format and old single-channel format for compatibility
        try:
            multi_channel_data, labels, channel_info = load_training_data(str(edf_path), str(xml_path))
            #I have added this:
            multi_channel_list.append(multi_channel_data)#Save the whole dictionary
            print(f"Multi-channel data loaded:")
            print(f"  EEG: {multi_channel_data['eeg'].shape}")
            print(f"Labels shape: {labels.shape}")

            print("multi_channel_data shape:", multi_channel_data['eeg'].shape)
            print("First 10 values of first epoch:", multi_channel_data['eeg'][0, :10])
            print("Labels shape", labels.shape)
            print("First 10 labels:", labels[:10])
            
            # For pipeline compatibility, use EEG data as primary signal
            eeg_data = multi_channel_data['eeg'][:, 0, :]  # Use first EEG channel for now
            print("unique values:", np.unique(eeg_data)[:10])
            print(f"Using EEG channel 1 for pipeline: {eeg_data.shape}")
            all_epochs.append(eeg_data)
            all_labels.append(labels)
            all_record_ids.extend([record_id] * len(labels)) 
      

        except (ValueError, TypeError):
            # Fallback to old format if multi-channel not implemented
            eeg_data, labels = load_training_data(edf_file, xml_path)
            print(f"Single-channel data loaded: {eeg_data.shape}, Labels: {labels.shape}")

#    combined_epochs = np.concatenate(all_epochs, axis=0)
#    combined_labels = np.concatenate(all_labels, axis=0)


    if config.CURRENT_ITERATION == 1:
        combined_epochs = np.concatenate(all_epochs, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
    else:
        eeg_all = [mc['eeg'] for mc in multi_channel_list]
        eog_all = [mc['eog'] for mc in multi_channel_list]
        combined_epochs = {
            'eeg': np.concatenate(eeg_all, axis=0),
            'eog': np.concatenate(eog_all, axis=0)
        }
        combined_labels = np.concatenate(all_labels, axis=0)


    # Print label distribution
    unique, counts = np.unique(combined_labels, return_counts=True) # finds all unique values in the combined_labels array and counts how many times each appears.
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    total = combined_labels.size

    print("\nLabel distribution across all data:")
    for stage, count in zip(stage_names, counts):
        percent = 100 * count / total
        print(f"  {stage}: {count} epochs ({percent:.1f}%)")


    #print("\nEpochs shape:", combined_epochs.shape)
    #print("Labels shape:", combined_labels.shape)

    #if combined_epochs.shape[0] != combined_labels.shape[0]:
    #    print("Warning: Number of epochs does not match number of labels!")
    #else:
    #    print("\nEpochs and labels are correctly aligned.")

    # Print shapes appropriately:
    if isinstance(combined_epochs, dict):
        print("\nEEG shape:", combined_epochs['eeg'].shape)
        print("EOG shape:", combined_epochs['eog'].shape)
        num_epochs = combined_epochs['eeg'].shape[0]
    else:
        print("\nEpochs shape:", combined_epochs.shape)
        num_epochs = combined_epochs.shape[0]

    print("Labels shape:", combined_labels.shape)

    # Alignment check:
    if num_epochs != combined_labels.shape[0]:
        print("Warning: Number of epochs does not match number of labels!")
    else:
        print("\nEpochs and labels are correctly aligned.")


    
    # 2. Preprocessing
    print("\n=== STEP 2: PREPROCESSING ===")
    preprocessed_data = None
    cache_filename_preprocess = f"preprocessed_data_iter{config.CURRENT_ITERATION}.joblib"

    print(f"Type of eeg_data: {type(combined_epochs)}")
    if isinstance(combined_epochs, dict):
        
        print(f"Keys in eeg_data: {list(combined_epochs.keys())}")
    else:
        print(f"eeg_data shape: {combined_epochs.shape}")



    if config.USE_CACHE:
        preprocessed_data = load_cache(cache_filename_preprocess, config.CACHE_DIR)
        if preprocessed_data is not None:
            print("Loaded preprocessed data from cache")

    if preprocessed_data is None:
        preprocessed_data = preprocess(combined_epochs, config, channel_info)
        # print("EEG preprocessed data shape:", preprocessed_data['eeg'].shape)
        # print("EOG preprocessed data shape:", preprocessed_data['eog'].shape)
        if config.USE_CACHE:
            save_cache(preprocessed_data, cache_filename_preprocess, config.CACHE_DIR)
            print("Saved preprocessed data to cache")


    # 3. Feature Extraction
    print("\n=== STEP 3: FEATURE EXTRACTION ===")
    features = None
    cache_filename_features = f"features_iter{config.CURRENT_ITERATION}.joblib"
    if config.USE_CACHE:
        features = load_cache(cache_filename_features, config.CACHE_DIR)
        if features is not None:
            print("Loaded features from cache")

    if features is None:
        features = extract_features(preprocessed_data, config,channel_info)
        print(f"Extracted features shape: {features.shape}")
        if features.shape[1] == 0:
            print("WARNING: No features extracted! Students must implement feature extraction.")
        if config.USE_CACHE:
            save_cache(features, cache_filename_features, config.CACHE_DIR)
            print("Saved features to cache")

#     # 4. Feature Selection
#     print("\n=== STEP 4: FEATURE SELECTION ===")
#     selected_features = select_features(features, labels, config)
#     print(f"Selected features shape: {selected_features.shape}")

#     # 5. Classification
#     print("\n=== STEP 5: CLASSIFICATION ===")
#     if selected_features.shape[1] > 0:
#         model = train_classifier(selected_features, labels, config)
#         print(f"Trained {config.CLASSIFIER_TYPE} classifier")
#     else:
#         print("⚠️  WARNING: Cannot train classifier - no features available!")
#         print("Students must implement feature extraction first.")
#         model = None

#     # 6. Visualization
#     print("\n=== STEP 6: VISUALIZATION ===")
#     if model is not None:
#         visualize_results(model, selected_features, labels, config)
#     else:
#         print("Skipping visualization - no trained model")

#     # # 7. Report Generation
#     # print("\n=== STEP 7: PROCESSING LOG & REPORT GENERATION ===")

#     # # Restore the original stdout
#     # #sys.stdout = original_stdout

#     # # Get the captured output from the buffer
#     # #processing_log = stdout_buffer.getvalue()   
     
#     # if model is not None:
#     #     #generate_report(model, selected_features, labels, config, processing_log)
#     #     generate_report(model, selected_features, labels, config, "")

#     # else:
#     #     print("Skipping report - no trained model")

#     # print("\n" + "="*50)
#     # print("PIPELINE FINISHED")
#     # if model is None:
#     #     print("⚠️  Students need to implement missing components!")
#     # print("="*50)

if __name__ == "__main__":
    main()