function run_inference()
%% Script to run inference on hold-out data and generate submission file.

clc;
close all;
clearvars -except config;

% Add src directory and subdirectories to path
addpath(genpath('src'));

% Load configuration
run('config.m'); % This will load config variables into the workspace

fprintf('--- Sleep Scoring Inference - Iteration %d ---\n', CURRENT_ITERATION);

% Load the trained model (assuming it was saved during training)
model_filename = sprintf('model_iter%d.mat', CURRENT_ITERATION);
model = load_cache(model_filename, CACHE_DIR);
if isempty(model)
    fprintf('Error: Trained model not found. Please run main.m first to train a model.\n');
    return;
end

% 1. Load Hold-out Data
% For jumpstart, we're using dummy data. In a real scenario, you'd iterate through files.
holdout_edf_file = fullfile(HOLDOUT_DIR, "dummy_holdout.edf"); % Placeholder
holdout_eeg_data = data_loader_load_holdout_data(holdout_edf_file);

% 2. Preprocessing (using the same logic as training)
preprocessed_holdout_data = [];
cache_filename_preprocess_holdout = sprintf('preprocessed_holdout_data_iter%d.mat', CURRENT_ITERATION);
if USE_CACHE
    preprocessed_holdout_data = load_cache(cache_filename_preprocess_holdout, CACHE_DIR);
end

if isempty(preprocessed_holdout_data)
    preprocessed_holdout_data = preprocessing_preprocess(holdout_eeg_data, config);
    if USE_CACHE
        save_cache(preprocessed_holdout_data, cache_filename_preprocess_holdout, CACHE_DIR);
    end
end

% 3. Feature Extraction (using the same logic as training)
holdout_features = [];
cache_filename_features_holdout = sprintf('features_holdout_iter%d.mat', CURRENT_ITERATION);
if USE_CACHE
    holdout_features = load_cache(cache_filename_features_holdout, CACHE_DIR);
end

if isempty(holdout_features)
    holdout_features = feature_extraction_extract_features(preprocessed_holdout_data, config);
    if USE_CACHE
        save_cache(holdout_features, cache_filename_features_holdout, CACHE_DIR);
    end
end

% 4. Make Inference
predictions = inference_make_inference(model, holdout_features, config);

% Dummy record and epoch numbers for submission file
record_numbers = ones(size(predictions)); % Assuming one record for dummy data
epoch_numbers = (0:length(predictions)-1)';

% 5. Generate Submission File
inference_generate_submission_file(predictions, record_numbers, epoch_numbers, config);

fprintf('--- Inference Finished ---\n');

end
