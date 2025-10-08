function main()
%% Main script to run the sleep scoring pipeline.

clc;
close all;
clearvars -except config;

% Add src directory and subdirectories to path
addpath(genpath('src'));

% Load configuration
run('config.m'); % This will load config variables into the workspace

fprintf('--- Sleep Scoring Pipeline - Iteration %d ---\n', CURRENT_ITERATION);

% 1. Load Data
% Example uses R1.edf and R1.xml - students should adapt for their dataset
edf_file = fullfile(SAMPLE_DIR, 'R1.edf'); % Example EDF file
xml_file = fullfile(SAMPLE_DIR, 'R1.xml'); % Corresponding annotation file

% Handle both multi-channel and single-channel formats
try
    [multi_channel_data, labels, channel_info] = load_training_data(edf_file, xml_file);
    fprintf('Multi-channel data loaded\n');
    % Use first EEG channel for pipeline compatibility
    eeg_data = squeeze(multi_channel_data.eeg(:, 1, :));
    fprintf('Using EEG channel 1 for pipeline\n');
catch
    % Fallback to old format
    [eeg_data, labels] = load_training_data(edf_file, xml_file);
    fprintf('Single-channel data loaded\n');
end

% 2. Preprocessing
preprocessed_data = [];
cache_filename_preprocess = sprintf('preprocessed_data_iter%d.mat', CURRENT_ITERATION);
if USE_CACHE
    preprocessed_data = load_cache(cache_filename_preprocess, CACHE_DIR);
end

if isempty(preprocessed_data)
    preprocessed_data = preprocess(eeg_data);
    if USE_CACHE
        save_cache(preprocessed_data, cache_filename_preprocess, CACHE_DIR);
    end
end

% 3. Feature Extraction
features = [];
cache_filename_features = sprintf('features_iter%d.mat', CURRENT_ITERATION);
if USE_CACHE
    features = load_cache(cache_filename_features, CACHE_DIR);
end

if isempty(features)
    features = extract_features(preprocessed_data);
    if USE_CACHE
        save_cache(features, cache_filename_features, CACHE_DIR);
    end
end

% 4. Feature Selection
selected_features = select_features(features, labels);

% 5. Classification
model = train_classifier(selected_features, labels);

% Save the trained model for inference
model_filename = sprintf('model_iter%d.mat', CURRENT_ITERATION);
save_cache(model, model_filename, CACHE_DIR);

% 6. Visualization
visualize_results(model, selected_features, labels);

% 7. Report Generation
generate_report(model, selected_features, labels);

fprintf('--- Pipeline Finished ---\n');

end
