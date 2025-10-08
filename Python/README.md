# Sleep Scoring Project - Python Jumpstart

This directory contains a jumpstart project for the automatic sleep scoring system, implemented in Python. It provides a modular, function-based design to help students quickly get started and develop further.

## 🚀 How to Use the Jumpstart

⚠️ **IMPORTANT**: This jumpstart provides **structure and examples only**. Students must implement the core algorithms themselves!

### 1. Setup

First, ensure you have Python 3.x installed. Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 1.1 Verify Setup

Run the tests to ensure everything is configured correctly:

```bash
python -m pytest tests/ -v
```

All tests should pass. If not, check your environment setup.

### 2. Project Structure

```
/Python/
├── cache/                  # Stores cached preprocessed data and extracted features
├── src/                    # Source code for different modules of the pipeline
│   ├── data_loader.py      # Handles loading EDF and XML files
│   ├── preprocessing.py    # Contains functions for signal preprocessing (e.g., filtering)
│   ├── feature_extraction.py # Extracts features from preprocessed data
│   ├── feature_selection.py # Selects relevant features (placeholder)
│   ├── classification.py   # Implements classification algorithms
│   ├── visualization.py    # For plotting results (e.g., confusion matrix)
│   ├── report.py           # Generates summary reports
│   ├── inference.py        # Handles making predictions on hold-out data
│   └── utils.py            # Utility functions (e.g., caching)
├── tests/                  # Unit tests for each module
├── main.py                 # Orchestrates the training and evaluation pipeline
├── run_inference.py        # Script to run inference on hold-out data and generate submission file
├── config.py               # Project configuration (iterations, file paths, model parameters)
├── requirements.txt        # List of Python dependencies
└── colab_notebook.ipynb    # Google Colab notebook for running the pipeline
```

### 3. Running the Training and Evaluation Pipeline

⚠️ **Expected Behavior**: The pipeline will run but show warnings about missing implementations.

To run the full pipeline for the current iteration defined in `config.py`:

```bash
python main.py
```

**First Run Results**:
- ✅ Configuration loads successfully
- ✅ Multi-channel dummy data is loaded (240 epochs, 2 hours):
  - 2 EEG channels (C3-A2, C4-A1) at 125 Hz (3750 samples/epoch)
  - 2 EOG channels (EOG(L), EOG(R)) at 50 Hz (1500 samples/epoch)
  - 1 EMG channel at 125 Hz (3750 samples/epoch)
- ✅ Basic preprocessing applies simple lowpass filters
- ⚠️ Feature extraction only produces 6 features (2 EEG channels × 3 features each)
- ⚠️ Students must implement 13+ additional time-domain features per EEG channel
- ⚠️ Feature selection is placeholder (students must implement)
- ✅ Basic k-NN classifier trains (but with minimal features)

**What Students Must Implement**:
1. **Multi-Channel Data Loading**:
   - Real EDF/XML file parsing for R1.edf/R1.xml format
   - Handle 2 EEG + 2 EOG + 1 EMG channels with different sampling rates
   - Channel identification by name patterns (C3-A2, C4-A1, EOG(L), EOG(R), EMG)
   - **Actual sampling rates**: EEG 125 Hz, EOG 50 Hz, EMG 125 Hz
2. **Multi-Channel Preprocessing**:
   - Different filtering for each signal type (EEG, EOG, EMG)
   - Cross-channel artifact detection and removal
   - Sampling rate harmonization or native-rate processing
   - Note: Hardware high-pass filters already applied (0.15 Hz for EEG/EOG/EMG)
3. **Comprehensive Feature Extraction**:
   - 13+ additional time-domain features per channel (Hjorth parameters, etc.)
   - Signal-specific features (eye movements for EOG, muscle tone for EMG)
   - Frequency-domain features (band powers, spectral features)
   - Cross-channel features (correlations, coherence)
4. **Feature Selection**: Statistical tests, mutual information, recursive elimination
5. **Advanced Classification**: Hyperparameter tuning, cross-validation, ensemble methods

### 4. Running Inference and Generating Submission File

After training your model using `main.py`, you can run inference on the hold-out data and generate a `submission.csv` file:

```bash
python run_inference.py
```

This will create a `submission.csv` file in the `data/` directory, formatted as `record_number,epoch_number,label`.

### 5. Configuration

The `config.py` file is central to managing the project. You can adjust:
*   `CURRENT_ITERATION`: To switch between different stages of development (1-4).
*   `USE_CACHE`: To enable/disable caching of intermediate results.
*   File paths, preprocessing parameters, and model hyperparameters.

### 6. Google Colab

Alternatively, you can run the pipeline in Google Colab:

1.  Open the `colab_notebook.ipynb` file in Google Colab.
2.  Follow the instructions in the notebook to clone the repository, install dependencies, and run the pipeline.

## 📖 Project Planning and Iterative Development

This project follows an agile, iterative development approach as outlined in the `PROJECT_GUIDE.md` file in the root directory. The `config.py` file allows you to set the `CURRENT_ITERATION` to control the behavior of the pipeline, enabling progressive development and testing of features.

## 💿 Data Information

### Data Structure

*   **`data/training/`**: This directory should contain EDF files and their corresponding XML annotation files for training and validation of your models. These files have associated sleep stage labels.
*   **`data/holdout/`**: This directory should contain EDF files for which you need to predict sleep stages. These files do *not* have associated labels and are used for competition submission.
*   **`data/sample/`**: This directory contains a small sample of data for testing the pipeline.

### EDF File Format

EDF (European Data Format) is a standard file format for storing physiological and biological signals. It can store multiple signals (e.g., EEG, EOG, EMG) and includes metadata like sampling frequency and channel names. Sleep stages are typically labeled for every 30-second epoch.

To read EDF files in Python, the `mne` library is recommended. A basic `read_edf` function is provided in `src/data_loader.py`.

### XML Annotation Files

XML (Extensible Markup Language) files are used to store structured data, including annotations for the EDF signals. These annotations typically contain the sleep stage labels for each epoch.

For more details on the Compumedics Annotation Format, refer to: [https://github.com/nsrr/edf-editor-translator/wiki/Compumedics-Annotation-Format](https://github.com/nsrr/edf-editor-translator/wiki/Compumedics-Annotation-Format)

## 🏆 Competition Submission

For the competition, you will submit a CSV file generated by `run_inference.py`. The format should be:

```csv
record_number,epoch_number,label
1,0,Wake
1,1,N1
...
```

`record_number` refers to the identifier of the EDF file, and `epoch_number` is the 0-indexed sequence of 30-second epochs within that record. `label` is the predicted sleep stage.