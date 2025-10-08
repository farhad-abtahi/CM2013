function [eeg_data, record_info] = load_holdout_data(edfFilePath)
%% STUDENT IMPLEMENTATION AREA: Load holdout EDF files (no labels).
%
% Similar to load_training_data but without XML annotations.
% Students must implement actual EDF loading for competition data.

fprintf('Loading holdout data from %s...\n', edfFilePath);

% TODO: Students must implement actual EDF loading
% DUMMY DATA for jumpstart testing - students must replace:
fprintf('WARNING: Using dummy data! Students must implement actual EDF loading.\n');
nEpochs = 960; % 8 hours of sleep recording
nSamples = 30 * 125; % 30 seconds at 125 Hz (actual EEG sampling rate)
eeg_data = randn(nEpochs, nSamples);
record_info = struct('record_id', 1, 'n_epochs', nEpochs); % Dummy metadata
fprintf('Generated dummy holdout data: %d epochs (%.1f hours)\n', nEpochs, nEpochs/120);

end
