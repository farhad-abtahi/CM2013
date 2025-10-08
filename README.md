# Sleep Scoring Project

Welcome to the Sleep Scoring Project! This repository contains a modular framework for developing an automatic sleep scoring system, available in both Python and MATLAB.

## 🚀 Project Overview

This project is designed to guide students through the process of building a biomedical signal processing pipeline. It emphasizes an agile, iterative development approach, allowing for progressive enhancement of the system's capabilities.

The goal is to classify sleep stages from biosignals (EEG, EOG, EMG) with a target accuracy of 75-80%. The project follows an agile development process with four iterations, progressively enhancing the system's capabilities.

## 📂 Repository Structure

```
/Users/sabt/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Github/CM2013/
├───.DS_Store
├───GEMINI.md
├───PROJECT_GUIDE.md
├───README.md
├───.git/...
├───data/
│   ├───holdout/
│   └───training/
│       ├───dummy.edf
│       └───dummy.xml
├───MATLAB/
│   ├───.DS_Store
│   ├───config.m
│   ├───main.m
│   ├───README.md
│   ├───run_inference.m
│   ├───cache/
│   ├───src/
│   │   ├───classification.m
│   │   ├───data_loader.m
│   │   ├───feature_extraction.m
│   │   ├───feature_selection.m
│   │   ├───inference.m
│   │   ├───preprocessing.m
│   │   ├───report.m
│   │   ├───visualization.m
│   │   └───utils/
│   │       ├───load_cache.m
│   │       └───save_cache.m
│   └───tests/
│       ├───test_data_loader.m
│       └───test_preprocessing.m
└───Python/
    ├───.coverage
    ├───config.py
    ├───main.py
    ├───README.md
    ├───requirements.txt
    ├───run_inference.py
    ├───__pycache__/...
    ├───.pytest_cache/...
    ├───cache/
    ├───src/
    │   ├───__init__.py
    │   ├───classification.py
    │   ├───data_loader.py
    │   ├───feature_extraction.py
    │   ├───feature_selection.py
    │   ├───inference.py
    │   ├───preprocessing.py
    │   ├───report.py
    │   ├───utils.py
    │   ├───visualization.py
    │   ├───__pycache__/...
    │   └───utils/
    └───tests/
        ├───__init__.py
        ├───test_data_loader.py
        ├───test_preprocessing.py
        └───__pycache__/...
```

## 📖 Guides

*   **[PROJECT_GUIDE.md](./PROJECT_GUIDE.md):** The main guide for the project. It covers project management, development timeline, system architecture, and the ClickUp guide.
*   **[Python/README.md](./Python/README.md):** A focused guide for the Python implementation, covering setup, running the code, and configuration.
*   **[MATLAB/README.md](./MATLAB/README.md):** A focused guide for the MATLAB implementation, covering setup, running the code, and configuration.

## 💿 Data

The `data/` directory at the root of this repository contains:

*   **`data/training/`**: Labeled EDF and XML files for training and validation.
*   **`data/holdout/`**: Unlabeled EDF files for final inference and competition submission.

## 🏆 Competition Submission

Students are expected to submit a CSV file with predicted sleep stages for the hold-out data. The format should be `record_number,epoch_number,label`.

Good luck with your project! 🚀