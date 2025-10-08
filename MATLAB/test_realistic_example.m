%% Sleep Scoring - Realistic EDF/XML Example
% This script demonstrates proper handling of EDF files with multiple sampling rates
% and visualization of sleep data including hypnogram
%
% STUDENTS: This is an EXAMPLE showing proper EDF/XML handling.
% You should understand this approach and adapt it for your pipeline.

clc;
close all;
clear all;

%% Load configuration
run('config.m');

fprintf('=== Realistic Multi-Channel EDF/XML Example ===\n');

% This example demonstrates proper handling of multi-channel EDF files
% with R1.edf and R1.xml as reference examples

%% EXAMPLE 1: Show how to read R1.edf with proper multi-channel handling
fprintf('\n--- Multi-Channel EDF File Structure Example ---\n');

edf_example = fullfile(SAMPLE_DIR, 'R1.edf');
xml_example = fullfile(SAMPLE_DIR, 'R1.xml');

if exist(edf_example, 'file')
    try
        % This would work with real EDF files using the provided edfread function
        fprintf('Reading EDF file structure...\n');

        % STUDENT TODO: Uncomment these lines when you have real EDF files:
        % [hdr, record] = edfread(fullfile(SAMPLE_DIR, 'dummy.edf'));
        % [events, stages, epochLength, annotation] = readXML(fullfile(SAMPLE_DIR, 'dummy.xml'));

        % For demonstration, show what the structure would look like:
        fprintf('EDF Header would contain:\n');
        fprintf('  - hdr.samples: sampling frequencies for each channel\n');
        fprintf('  - hdr.label: channel names (EEG, EOG, EMG, etc.)\n');
        fprintf('  - hdr.duration: recording length\n');
        fprintf('  - hdr.ns: number of signals\n');

        fprintf('Typical sampling frequencies:\n');
        fprintf('  - EEG channels: 256 Hz\n');
        fprintf('  - EOG channels: 256 Hz\n');
        fprintf('  - EMG channels: 256 Hz\n');
        fprintf('  - ECG channels: 512 Hz\n');

    catch ME
        fprintf('Note: Real EDF reading requires actual files. Error: %s\n', ME.message);
    end
else
    fprintf('No real EDF file found - using dummy data structure\n');
end

%% EXAMPLE 2: Demonstrate proper epoch extraction with multiple sampling rates
fprintf('\n--- Multi-Rate Epoch Extraction Example ---\n');

% Simulate realistic EDF structure
n_channels = 4;
channel_names = {'EEG_C3', 'EEG_C4', 'EOG_L', 'EMG_Chin'};
sampling_rates = [256, 256, 256, 512];  % Different rates for different signals
recording_duration_sec = 8 * 3600;  % 8 hours

fprintf('Simulated EDF structure:\n');
for i = 1:n_channels
    samples_total = recording_duration_sec * sampling_rates(i);
    epochs_30sec = floor(recording_duration_sec / 30);
    samples_per_epoch = 30 * sampling_rates(i);

    fprintf('  %s: %d Hz, %d samples/epoch, %d total epochs\n', ...
            channel_names{i}, sampling_rates(i), samples_per_epoch, epochs_30sec);
end

%% EXAMPLE 3: Demonstrate proper hypnogram visualization
fprintf('\n--- Hypnogram Visualization Example ---\n');

% Generate realistic sleep stage progression (8 hours = 960 epochs)
n_epochs = 960;
time_hours = (1:n_epochs) * 30 / 3600;  % Convert epochs to hours

% Create realistic sleep progression
stages = generate_realistic_sleep_stages(n_epochs);

% Plot hypnogram
figure('Position', [100, 100, 800, 400]);
plot(time_hours, stages, 'LineWidth', 2);
ylim([0 5]);
xlim([0 8]);
set(gca, 'YTick', [0:4], 'YTickLabel', {'REM', 'N3', 'N2', 'N1', 'Wake'});
xlabel('Time (Hours)');
ylabel('Sleep Stage');
title('Example Hypnogram - 8 Hour Sleep Study');
grid on;
set(gcf, 'Color', 'w');

% Add sleep statistics
wake_pct = sum(stages == 4) / n_epochs * 100;
n1_pct = sum(stages == 3) / n_epochs * 100;
n2_pct = sum(stages == 2) / n_epochs * 100;
n3_pct = sum(stages == 1) / n_epochs * 100;
rem_pct = sum(stages == 0) / n_epochs * 100;

fprintf('Sleep Stage Distribution:\n');
fprintf('  Wake: %.1f%%\n', wake_pct);
fprintf('  N1: %.1f%%\n', n1_pct);
fprintf('  N2: %.1f%%\n', n2_pct);
fprintf('  N3: %.1f%%\n', n3_pct);
fprintf('  REM: %.1f%%\n', rem_pct);

%% EXAMPLE 4: Show multi-channel signal visualization
fprintf('\n--- Multi-Channel Signal Visualization ---\n');

% Simulate one 30-second epoch from multiple channels
epoch_duration = 30;  % seconds
epoch_num = 100;  % Show epoch 100

figure('Position', [100, 500, 1000, 600]);
for ch = 1:n_channels
    fs = sampling_rates(ch);
    n_samples = epoch_duration * fs;
    t = (0:n_samples-1) / fs;

    % Generate realistic-looking signal (filtered noise)
    signal = generate_realistic_signal(channel_names{ch}, n_samples, fs);

    subplot(n_channels, 1, ch);
    plot(t, signal, 'LineWidth', 0.8);
    ylabel(sprintf('%s\n(µV)', channel_names{ch}));
    xlim([0 30]);
    grid on;

    if ch == 1
        title(sprintf('30-second Epoch #%d - Multi-Channel View', epoch_num));
    end
    if ch == n_channels
        xlabel('Time (seconds)');
    end
end
set(gcf, 'Color', 'w');

%% STUDENT GUIDANCE
fprintf('\n=== STUDENT IMPLEMENTATION GUIDANCE ===\n');
fprintf('1. Use edfread() to load real EDF files with proper sampling rates\n');
fprintf('2. Use readXML() to load sleep stage annotations\n');
fprintf('3. Handle different sampling rates per channel:\n');
fprintf('   - Resample all to common rate (e.g., 256 Hz), OR\n');
fprintf('   - Extract features at native rates, then combine\n');
fprintf('4. Implement proper epoch extraction (30-second windows)\n');
fprintf('5. Create visualizations for data validation:\n');
fprintf('   - Hypnogram for sleep stage progression\n');
fprintf('   - Multi-channel signal plots for quality check\n');
fprintf('6. Handle missing data, artifacts, and edge cases\n');

fprintf('\nExample complete! Students should study this structure.\n');

%% Helper Functions
function stages = generate_realistic_sleep_stages(n_epochs)
    % Generate realistic sleep stage progression
    stages = zeros(n_epochs, 1);

    % Initial wake period
    stages(1:20) = 4;  % Wake

    % Sleep onset
    stages(21:40) = 3;  % N1
    stages(41:100) = 2;  % N2

    % First deep sleep
    stages(101:200) = 1;  % N3

    % Cycling pattern (simplified)
    for cycle_start = 201:200:n_epochs-200
        cycle_end = min(cycle_start + 199, n_epochs);
        cycle_length = cycle_end - cycle_start + 1;

        % Each cycle: N2 -> N3 -> N2 -> REM
        n2_1 = round(cycle_length * 0.3);
        n3_period = round(cycle_length * 0.4);
        n2_2 = round(cycle_length * 0.2);
        rem_period = cycle_length - n2_1 - n3_period - n2_2;

        idx = cycle_start;
        stages(idx:idx+n2_1-1) = 2;
        idx = idx + n2_1;
        stages(idx:idx+n3_period-1) = 1;
        idx = idx + n3_period;
        stages(idx:idx+n2_2-1) = 2;
        idx = idx + n2_2;
        if idx <= cycle_end
            stages(idx:cycle_end) = 0;  % REM
        end
    end

    % Final wake period
    if n_epochs > 900
        stages(900:end) = 4;  % Wake before morning
    end
end

function signal = generate_realistic_signal(channel_name, n_samples, fs)
    % Generate realistic-looking biosignal
    t = (0:n_samples-1) / fs;

    switch lower(channel_name)
        case {'eeg_c3', 'eeg_c4'}
            % EEG: alpha (8-13Hz), theta (4-8Hz), some higher frequencies
            signal = 20 * sin(2*pi*10*t) + 15 * sin(2*pi*6*t) + ...
                     10 * randn(size(t)) + 5 * sin(2*pi*25*t);
        case 'eog_l'
            % EOG: slower eye movements
            signal = 50 * sin(2*pi*0.5*t) + 20 * randn(size(t));
        case 'emg_chin'
            % EMG: higher frequency muscle activity
            signal = 30 * randn(size(t)) + 10 * sin(2*pi*50*t);
        otherwise
            signal = 10 * randn(size(t));
    end

    % Add some realistic amplitude scaling
    signal = signal * (50 + 20*randn());  % µV scale
end