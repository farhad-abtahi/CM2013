function plot_sample_epoch(edf_path, epoch_idx, epoch_duration)
%% Plot all signals from a sample epoch in an EDF file with correct labels and sampling frequencies.
%
% Usage:
%   plot_sample_epoch('data/training/file.edf', 0)  % Plot first epoch
%   plot_sample_epoch('data/training/file.edf', 100, 30)  % Plot epoch 100
%
% Args:
%   edf_path: Path to EDF file
%   epoch_idx: Index of epoch to plot (default: 0)
%   epoch_duration: Duration of each epoch in seconds (default: 30)

if nargin < 2
    epoch_idx = 0;
end
if nargin < 3
    epoch_duration = 30;
end

try
    % Read EDF file
    [hdr, record] = edfread(edf_path);

    n_channels = length(hdr.label);

    % Calculate epoch boundaries
    start_time = epoch_idx * epoch_duration;

    fprintf('\nPlotting Epoch %d (Time: %d-%ds)\n', epoch_idx, start_time, start_time+epoch_duration);
    fprintf('%s\n', repmat('=', 1, 70));

    % Create figure
    figure('Position', [100 100 1200 800]);

    for ch_idx = 1:n_channels
        fs = hdr.frequency(ch_idx);  % Sampling frequency
        label = hdr.label{ch_idx};   % Channel label

        % Calculate sample indices for this epoch
        start_sample = floor(start_time * fs) + 1;
        n_samples = floor(epoch_duration * fs);
        end_sample = min(start_sample + n_samples - 1, length(record(ch_idx, :)));

        % Extract signal data for this epoch
        signal = record(ch_idx, start_sample:end_sample);

        % Create time axis in seconds
        time = (0:length(signal)-1) / fs + start_time;

        % Create subplot
        subplot(n_channels, 1, ch_idx);
        plot(time, signal, 'LineWidth', 0.5);
        ylabel(sprintf('%s\n(%d Hz)', label, fs), 'FontSize', 10);
        grid on;
        xlim([start_time, start_time + epoch_duration]);

        if ch_idx == 1
            title(sprintf('Sleep Signals - Epoch %d (%ds window)', epoch_idx, epoch_duration), ...
                  'FontSize', 14, 'FontWeight', 'bold');
        end

        if ch_idx == n_channels
            xlabel('Time (seconds)', 'FontSize', 12);
        end

        fprintf('  %s: %d Hz, %d samples\n', label, fs, length(signal));
    end

catch ME
    if strcmp(ME.identifier, 'MATLAB:UndefinedFunction')
        fprintf('Error: edfread function not found.\n');
        fprintf('Please install the Biosig toolbox or use a custom EDF reader.\n');
        fprintf('You can download Biosig from: http://biosig.sourceforge.net/\n');
    else
        fprintf('Error reading EDF file: %s\n', ME.message);
    end
end

end
