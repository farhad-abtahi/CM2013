# -- Project Configuration --

# Set the current iteration of the project (1-4). 
# This controls which parts of the pipeline are active.
CURRENT_ITERATION = 1

# Set to True to use cached data for preprocessing and feature extraction.
USE_CACHE = False  # Temporarily disabled for testing with real data

# -- File Paths --
import os
DATA_DIR = '../data/'
TRAINING_DIR = f'{DATA_DIR}training/'
HOLDOUT_DIR = f'{DATA_DIR}holdout/'
SAMPLE_DIR = f'{DATA_DIR}sample/'
CACHE_DIR = 'cache/'

# Validate and create directories if needed
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}\nPlease ensure you are running from the correct directory.")
if not os.path.exists(CACHE_DIR):
    print(f"Creating cache directory: {CACHE_DIR}")
    os.makedirs(CACHE_DIR, exist_ok=True)

# -- Preprocessing --
LOW_PASS_FILTER_FREQ = 40  # Hz

# -- Feature Extraction --
# (Add feature-specific parameters here)

# -- Classification --
# Iteration-specific parameters - students should modify these based on current iteration
if CURRENT_ITERATION == 1:
    # Iteration 1: Basic pipeline with k-NN
    CLASSIFIER_TYPE = 'knn'
    KNN_N_NEIGHBORS = 5
elif CURRENT_ITERATION == 2:
    # Iteration 2: Enhanced EEG processing with SVM
    CLASSIFIER_TYPE = 'svm'
    SVM_C = 1.0
    SVM_KERNEL = 'rbf'
elif CURRENT_ITERATION == 3:
    # Iteration 3: Multi-signal processing with Random Forest
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
elif CURRENT_ITERATION == 4:
    # Iteration 4: Full system optimization
    CLASSIFIER_TYPE = 'random_forest'
    RF_N_ESTIMATORS = 200
    RF_MAX_DEPTH = None
    RF_MIN_SAMPLES_SPLIT = 5
else:
    raise ValueError(f"Invalid CURRENT_ITERATION: {CURRENT_ITERATION}. Must be 1-4.")

# -- Submission --
SUBMISSION_FILE = 'submission.csv'