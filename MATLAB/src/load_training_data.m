function [eeg_data, labels] = load_training_data(edfFilePath, xmlFilePath)
%% STUDENT IMPLEMENTATION AREA: Load EDF and XML files.
%
% This function currently returns DUMMY DATA for jumpstart testing.
% Students must implement actual EDF/XML loading:
%
% 1. Load EDF file using read_edf function
% 2. Load XML annotations (sleep stage labels)
% 3. Extract relevant channels (EEG, EOG, EMG)
% 4. Segment into 30-second epochs
% 5. Handle different sampling rates
% 6. Match epochs with sleep stage labels

fprintf('Loading training data from %s and %s...\n', edfFilePath, xmlFilePath);

% TODO: Students must implement actual file loading
% DUMMY DATA for jumpstart testing - students must replace this:
fprintf('WARNING: Using dummy data! Students must implement actual EDF/XML loading.\n');

% NOTE FOR STUDENTS: This study has specific sampling rates:
% - EEG (C3-A2, C4-A1): 125 Hz
% - EOG (Left, Right): 50 Hz
% - EMG: 125 Hz
% Students must handle different sampling rates when loading real EDF files

% Realistic size: 8 hours = 8 * 60 * 2 = 960 epochs (30-second windows)
nEpochs = 960; % 8 hours of sleep recording
nSamples = 30 * 125; % 30 seconds at 125 Hz (actual EEG sampling rate)

% Generate realistic dummy data
eeg_data = randn(nEpochs, nSamples);

% Generate realistic sleep stage distribution (not uniform)
% Typical distribution: More N2, less N1 and REM, some Wake
stage_probs = [0.05, 0.05, 0.50, 0.25, 0.15]; % Wake, N1, N2, N3, REM
labels = randsample(0:4, nEpochs, true, stage_probs);

fprintf('Generated dummy sleep data: %d epochs (%.1f hours)\n', nEpochs, nEpochs/120);
for stage = 0:4
    count = sum(labels == stage);
    stage_names = {'Wake', 'N1', 'N2', 'N3', 'REM'};
    fprintf('  %s: %d epochs (%.1f%%)\n', stage_names{stage+1}, count, count/nEpochs*100);
end

end
