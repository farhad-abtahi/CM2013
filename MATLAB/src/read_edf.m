function [hdr, record] = read_edf(filePath)
%% EXAMPLE: Read an EDF file.
%
% This is a basic example. Students should expand this to:
% - Handle different EDF variants
% - Validate channel names and sampling rates
% - Handle missing or corrupted data
% - Extract specific time ranges

fprintf('Reading EDF file: %s\n', filePath);
% Placeholder for actual edfread call
% [hdr, record] = edfread(filePath);

% Dummy output for now - using ACTUAL sampling rates from study
hdr.samples = [125, 50, 125]; % Actual rates: EEG 125 Hz, EOG 50 Hz, EMG 125 Hz
hdr.label = {'C3-A2', 'EOG(L)', 'EMG'};
record = randn(3, 20*30*125); % Example dummy record (using highest rate)

end
