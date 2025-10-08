% -- Project Configuration --

% Set the current iteration of the project (1-4).
% This controls which parts of the pipeline are active.
CURRENT_ITERATION = 1;

% Set to true to use cached data for preprocessing and feature extraction.
USE_CACHE = true;

% -- File Paths --
DATA_DIR = '../data/';
TRAINING_DIR = [DATA_DIR 'training/'];
HOLDOUT_DIR = [DATA_DIR 'holdout/'];
SAMPLE_DIR = [DATA_DIR 'sample/'];
CACHE_DIR = 'cache/';

% Validate and create directories if needed
if ~exist(DATA_DIR, 'dir')
    error('Data directory not found: %s\nPlease ensure you are running from the correct directory.', DATA_DIR);
end
if ~exist(CACHE_DIR, 'dir')
    fprintf('Creating cache directory: %s\n', CACHE_DIR);
    mkdir(CACHE_DIR);
end

% -- Preprocessing --
LOW_PASS_FILTER_FREQ = 40; % Hz

% -- Feature Extraction --
% (Add feature-specific parameters here)

% -- Classification --
% Iteration-specific parameters - students should modify these based on current iteration
if CURRENT_ITERATION == 1
    % Iteration 1: Basic pipeline with k-NN
    CLASSIFIER_TYPE = 'knn';
    KNN_N_NEIGHBORS = 5;
elseif CURRENT_ITERATION == 2
    % Iteration 2: Enhanced EEG processing with SVM
    CLASSIFIER_TYPE = 'svm';
    SVM_C = 1.0;
    SVM_KERNEL = 'rbf';
elseif CURRENT_ITERATION == 3
    % Iteration 3: Multi-signal processing with Random Forest (TreeBagger in MATLAB)
    CLASSIFIER_TYPE = 'random_forest';
    RF_N_ESTIMATORS = 100;
    RF_MAX_DEPTH = 10;
elseif CURRENT_ITERATION == 4
    % Iteration 4: Full system optimization
    CLASSIFIER_TYPE = 'random_forest';
    RF_N_ESTIMATORS = 200;
    RF_MAX_DEPTH = []; % No limit
    RF_MIN_SAMPLES_SPLIT = 5;
else
    error('Invalid CURRENT_ITERATION: %d. Must be 1-4.', CURRENT_ITERATION);
end

% -- Submission --
SUBMISSION_FILE = 'submission.csv';
