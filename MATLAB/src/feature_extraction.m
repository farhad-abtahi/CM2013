function features = extract_features(data)
%% STUDENT IMPLEMENTATION AREA: Extract features based on current iteration.
%
% This function should handle both single-channel (old format) and
% multi-channel data (new format with 2 EEG + 2 EOG + 1 EMG channels).
%
% Iteration 1: 16 time-domain features per EEG channel
% Iteration 2: 31+ features (time + frequency domain) per channel
% Iteration 3: Multi-signal features (EEG + EOG + EMG)
% Iteration 4: Optimized feature set (selected subset)

% Get CURRENT_ITERATION from caller's workspace
try
    CURRENT_ITERATION = evalin('caller', 'CURRENT_ITERATION');
catch
    CURRENT_ITERATION = 1; % Default
end

fprintf('Extracting features for iteration %d...\n', CURRENT_ITERATION);

% Detect if we have multi-channel data structure
if isstruct(data) && isfield(data, 'eeg')
    fprintf('Processing multi-channel data (EEG + EOG + EMG)\n');
    features = extract_multi_channel_features(data, CURRENT_ITERATION);
else
    fprintf('Processing single-channel data (backward compatibility)\n');
    features = extract_single_channel_features(data, CURRENT_ITERATION);
end

end


function features = extract_multi_channel_features(multi_channel_data, CURRENT_ITERATION)
%% Extract features from multi-channel data: 2 EEG + 2 EOG + 1 EMG channels.
% Students should expand this significantly!

n_epochs = size(multi_channel_data.eeg, 1);
all_features = [];

for epoch_idx = 1:n_epochs
    epoch_features = [];

    % EEG features (2 channels)
    for ch = 1:size(multi_channel_data.eeg, 2)
        eeg_signal = squeeze(multi_channel_data.eeg(epoch_idx, ch, :));
        eeg_features = extract_time_domain_features(eeg_signal);
        epoch_features = [epoch_features, eeg_features];
    end

    if CURRENT_ITERATION >= 3
        % Add EOG features (2 channels)
        for ch = 1:size(multi_channel_data.eog, 2)
            eog_signal = squeeze(multi_channel_data.eog(epoch_idx, ch, :));
            eog_features = extract_eog_features(eog_signal);
            epoch_features = [epoch_features, eog_features];
        end

        % Add EMG features (1 channel)
        emg_signal = squeeze(multi_channel_data.emg(epoch_idx, 1, :));
        emg_features = extract_emg_features(emg_signal);
        epoch_features = [epoch_features, emg_features];
    end

    all_features = [all_features; epoch_features];
end

features = all_features;

if CURRENT_ITERATION == 1
    expected = 2 * 3; % 2 EEG channels × 3 features each
    fprintf('Multi-channel Iteration 1: %d features (target: %d+)\n', size(features, 2), expected);
    fprintf('Students must implement remaining 13 time-domain features per EEG channel!\n');
elseif CURRENT_ITERATION >= 3
    fprintf('Multi-channel features extracted: %d total\n', size(features, 2));
    fprintf('(2 EEG + 2 EOG + 1 EMG channels)\n');
end

end


function features = extract_single_channel_features(data, CURRENT_ITERATION)
%% Backward compatibility for single-channel data.

if CURRENT_ITERATION == 1
    % Iteration 1: Time-domain features (TARGET: 16 features)
    % CURRENT: Only 3 features implemented - students must add 13 more!
    all_features = [];
    for epoch_idx = 1:size(data, 1)
        epoch = data(epoch_idx, :);
        epoch_features = extract_time_domain_features(epoch);
        all_features = [all_features; epoch_features];
    end
    features = all_features;

    fprintf('WARNING: Only %d features extracted, target is 16 for iteration 1\n', size(features, 2));
    fprintf('Students must implement the remaining time-domain features!\n');

elseif CURRENT_ITERATION == 2
    % TODO: Students must implement frequency-domain features
    fprintf('TODO: Students must implement frequency-domain feature extraction\n');
    fprintf('Target: ~31 features (time + frequency domain)\n');
    features = zeros(size(data, 1), 0); % Empty features - students must implement

elseif CURRENT_ITERATION >= 3
    % TODO: Students must implement multi-signal features
    fprintf('TODO: Students should use multi-channel data format for iteration 3+\n');
    features = zeros(size(data, 1), 0); % Empty features - students must implement

else
    error('Invalid iteration: %d', CURRENT_ITERATION);
end

end


function features = extract_time_domain_features(epoch)
%% EXAMPLE: Extract basic time-domain features from a single epoch.
%
% This is a MINIMAL example with only 3 features.
% Students must implement the remaining 13+ time-domain features.
%
% Works for any signal type (EEG, EOG, EMG) but students should consider
% signal-specific features for optimal performance.

% EXAMPLE: Only 3 basic features - students must add 13+ more
features = [
    mean(epoch),    % Mean
    median(epoch),  % Median
    std(epoch)      % Standard deviation
];

% TODO: Students must implement remaining time-domain features:
% Basic statistical features:
% - var(epoch)              % Variance
% - rms(epoch)              % RMS
% - min(epoch)              % Minimum
% - max(epoch)              % Maximum
% - range(epoch)            % Range
% - skewness(epoch)         % Skewness
% - kurtosis(epoch)         % Kurtosis

% Signal complexity features:
% - zero_crossings(epoch)   % Zero crossings
% - hjorth_activity(epoch)  % Hjorth activity
% - hjorth_mobility(epoch)  % Hjorth mobility
% - hjorth_complexity(epoch)% Hjorth complexity

% Signal energy and power:
% - sum(epoch.^2)           % Total energy
% - mean(epoch.^2)          % Mean power

end


function features = extract_eog_features(eog_signal)
%% STUDENT TODO: Extract EOG-specific features for eye movement detection.
%
% EOG signals are used to detect:
% - Rapid eye movements (REM sleep indicator)
% - Slow eye movements
% - Eye blinks and artifacts

features = [
    mean(eog_signal),                           % Mean
    std(eog_signal),                           % Standard deviation
    max(eog_signal) - min(eog_signal)          % Range
];

% TODO: Students should add:
% - Eye movement detection features
% - Rapid vs slow movement discrimination
% - Cross-channel correlations (left vs right eye)

end


function features = extract_emg_features(emg_signal)
%% STUDENT TODO: Extract EMG-specific features for muscle tone detection.
%
% EMG signals are used to detect:
% - Muscle tone levels (high in wake, low in REM)
% - Muscle twitches and artifacts
% - Sleep-related muscle activity

features = [
    mean(emg_signal),                          % Mean
    std(emg_signal),                          % Standard deviation
    rms(emg_signal)                           % RMS
];

% TODO: Students should add:
% - High-frequency power (muscle activity indicator)
% - Spectral edge frequency
% - Muscle tone quantification

end


function rms_val = rms(signal)
%% Helper function: Root Mean Square
rms_val = sqrt(mean(signal.^2));
end