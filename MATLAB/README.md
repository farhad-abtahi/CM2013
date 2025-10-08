# Sleep Scoring Project - MATLAB Jumpstart

This directory contains a jumpstart project for the automatic sleep scoring system, implemented in MATLAB. It provides a modular, function-based design to help students quickly get started and develop further.

## 🚀 How to Use the Jumpstart

### 1. Setup

Ensure you have MATLAB installed. No specific toolbox installations are strictly required for the basic jumpstart, but the Signal Processing Toolbox and Statistics and Machine Learning Toolbox will be beneficial for further development.

### 2. Project Structure

```
/MATLAB/
├── cache/                  % Stores cached preprocessed data and extracted features
├── src/                    % Source code for different modules of the pipeline
│   ├── data_loader.m       % Handles loading EDF and XML files
│   ├── preprocessing.m     % Contains functions for signal preprocessing (e.g., filtering)
│   ├── feature_extraction.m % Extracts features from preprocessed data
│   ├── feature_selection.m % Selects relevant features (placeholder)
│   ├── classification.m    % Implements classification algorithms
│   ├── visualization.m     % For plotting results (e.g., confusion matrix)
│   ├── report.m            % Generates summary reports
│   ├── inference.m         % Handles making predictions on hold-out data
│   └── utils/              % Utility functions (e.g., caching)
├── tests/                  % Unit tests for each module (optional, can be implemented using MATLAB's testing framework)
├── main.m                  % Orchestrates the training and evaluation pipeline
├── run_inference.m         % Script to run inference on hold-out data and generate submission file
└── config.m                % Project configuration (iterations, file paths, model parameters)
```

### 3. Running the Training and Evaluation Pipeline

To run the full pipeline (data loading, preprocessing, feature extraction, classification, visualization, and reporting) for the current iteration defined in `config.m`:

1.  Open MATLAB.
2.  Navigate to the `/MATLAB/` directory.
3.  Run the `main.m` script from the MATLAB command window:
    ```matlab
    main
    ```

**Note:** The jumpstart code uses dummy data with correct sampling rates:
- EEG: 125 Hz (3,750 samples per 30-second epoch)
- EOG: 50 Hz (1,500 samples per 30-second epoch)
- EMG: 125 Hz (3,750 samples per 30-second epoch)

Students must implement real EDF/XML loading to use actual data files.

### 4. Running Inference and Generating Submission File

After training your model using `main.m`, you can run inference on the hold-out data and generate a `submission.csv` file:

1.  Open MATLAB.
2.  Navigate to the `/MATLAB/` directory.
3.  Run the `run_inference.m` script from the MATLAB command window:
    ```matlab
    run_inference
    ```

This will create a `submission.csv` file in the `data/` directory, formatted as `record_number,epoch_number,label`.

### 5. Configuration

The `config.m` file is central to managing the project. You can adjust:
*   `CURRENT_ITERATION`: To switch between different stages of development (1-4).
*   `USE_CACHE`: To enable/disable caching of intermediate results.
*   File paths, preprocessing parameters, and model hyperparameters.

## 📖 Project Planning and Iterative Development

This project follows an agile, iterative development approach as outlined in the `PROJECT_GUIDE.md` file in the root directory. The `config.m` file allows you to set the `CURRENT_ITERATION` to control the behavior of the pipeline, enabling progressive development and testing of features.

## 💿 Data Information

### Data Structure

*   **`data/training/`**: This directory should contain EDF files and their corresponding XML annotation files for training and validation of your models. These files have associated sleep stage labels.
*   **`data/holdout/`**: This directory should contain EDF files for which you need to predict sleep stages. These files do *not* have associated labels and are used for competition submission.

### EDF File Format

EDF (European Data Format) is a standard file format for storing physiological and biological signals. It can store multiple signals (e.g., EEG, EOG, EMG) and includes metadata like sampling frequency and channel names. Sleep stages are typically labeled for every 30-second epoch.

**Actual Sampling Rates in Study Data:**
- EEG (C3-A2, C4-A1): 125 Hz
- EOG (EOG(L), EOG(R)): 50 Hz
- EMG: 125 Hz
- ECG: 125 Hz
- Respiration (Thor/Abdo): 10 Hz
- SpO2/Heart Rate: 1 Hz

**Hardware Filters Already Applied:**
- EEG/EOG/EMG/ECG: High-pass 0.15 Hz
- Respiration: High-pass 0.05 Hz

To read EDF files in MATLAB, the `edfread` function is available. **Note:** The provided EDF files might not be compatible with the built-in `edfread` in MATLAB. You should use the `edfread` function provided in this module (if available) or a custom implementation.

### XML Annotation Files

XML (Extensible Markup Language) files are used to store structured data, including annotations for the EDF signals. These annotations typically contain the sleep stage labels for each epoch.

For more details on the Compumedics Annotation Format, refer to: [https://github.com/nsrr/edf-editor-translator/wiki/Compumedics-Annotation-Format](https://github.com/nsrr/edf-editor-translator/wiki/Compumedics-Annotation-Format)

## 🏆 Competition Submission

For the competition, you will submit a CSV file generated by `run_inference.m`. The format should be:

```csv
record_number,epoch_number,label
1,0,Wake
1,1,N1
...
```

`record_number` refers to the identifier of the EDF file, and `epoch_number` is the 0-indexed sequence of 30-second epochs within that record. `label` is the predicted sleep stage.