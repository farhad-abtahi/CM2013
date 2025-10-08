function plot_hypnogram(xml_path)
%% Plot hypnogram (sleep stage progression) from XML annotations.
%
% Usage:
%   plot_hypnogram('data/training/file.xml')
%
% Args:
%   xml_path: Path to XML annotation file

try
    % Parse XML file
    xml_doc = xmlread(xml_path);

    % Get all ScoredEvent elements
    scored_events = xml_doc.getElementsByTagName('ScoredEvent');

    epochs = [];
    stages = [];

    % Extract sleep stages and times
    for i = 0:scored_events.getLength()-1
        event = scored_events.item(i);

        % Get EventType
        event_type_nodes = event.getElementsByTagName('EventType');
        if event_type_nodes.getLength() > 0
            event_type = char(event_type_nodes.item(0).getFirstChild().getData());

            % Check if this is a sleep stage event
            if contains(event_type, 'Stage') || contains(event_type, 'Sleep')
                % Get EventConcept (stage name)
                event_concept_nodes = event.getElementsByTagName('EventConcept');
                start_nodes = event.getElementsByTagName('Start');
                duration_nodes = event.getElementsByTagName('Duration');

                if event_concept_nodes.getLength() > 0 && start_nodes.getLength() > 0
                    stage_name = char(event_concept_nodes.item(0).getFirstChild().getData());
                    start_time = str2double(char(start_nodes.item(0).getFirstChild().getData()));

                    if duration_nodes.getLength() > 0
                        dur = str2double(char(duration_nodes.item(0).getFirstChild().getData()));
                    else
                        dur = 30.0;
                    end

                    % Convert to epoch number
                    epoch_num = start_time / 30.0;
                    epochs = [epochs; epoch_num];

                    % Map stage names to numeric labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
                    stage_label = parse_stage_name(stage_name);
                    if ~isnan(stage_label)
                        stages = [stages; stage_label];
                    end
                end
            end
        end
    end

    if isempty(epochs)
        fprintf('Warning: No sleep stage annotations found in XML file\n');
        fprintf('The XML file may be in a different format or empty\n');
        return;
    end

    % Create hypnogram plot
    figure('Position', [100 100 1200 400]);

    % Plot as step function
    stage_names = {'Wake', 'N1', 'N2', 'N3', 'REM'};
    stage_colors = [1 0 0; 1 0.5 0; 0 0.8 0; 0 0 1; 0.5 0 0.5];  % RGB colors

    hold on;
    for i = 1:length(epochs)-1
        stage_idx = stages(i);
        plot([epochs(i), epochs(i+1)], [stage_idx, stage_idx], ...
             'Color', stage_colors(stage_idx+1, :), 'LineWidth', 2);
    end
    hold off;

    % Styling
    set(gca, 'YTick', 0:4);
    set(gca, 'YTickLabel', stage_names);
    ylabel('Sleep Stage', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('Epoch Number (30s epochs)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Hypnogram - Sleep Stage Progression', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    ylim([-0.5, 4.5]);

    % Add time axis on top
    ax1 = gca;
    ax2 = axes('Position', ax1.Position, 'XAxisLocation', 'top', 'YAxisLocation', 'right', 'Color', 'none');
    ax2.XLim = ax1.XLim;
    ax2.YLim = ax1.YLim;
    ax2.YTick = [];

    % Convert epochs to hours
    max_epoch = floor(epochs(end));
    hour_ticks = 0:120:max_epoch;  % 120 epochs = 1 hour
    ax2.XTick = hour_ticks;
    ax2.XTickLabel = arrayfun(@(x) sprintf('%.1f', x/120), hour_ticks, 'UniformOutput', false);
    xlabel(ax2, 'Time (hours)', 'FontSize', 12, 'FontWeight', 'bold');

    % Print statistics
    fprintf('\nSleep Stage Statistics:\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Total epochs: %d\n', length(stages));
    fprintf('Total duration: %.2f hours\n', length(stages)*30/3600);
    fprintf('\nStage distribution:\n');

    for stage_idx = 0:4
        count = sum(stages == stage_idx);
        percentage = count / length(stages) * 100;
        fprintf('  %s: %d epochs (%.1f%%)\n', stage_names{stage_idx+1}, count, percentage);
    end

catch ME
    fprintf('Error reading XML file: %s\n', ME.message);
    fprintf('Stack trace:\n');
    disp(getReport(ME));
end

end


function stage_label = parse_stage_name(stage_name)
%% Parse stage name and convert to numeric label (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)

stage_label = NaN;

% Create mapping
if contains(stage_name, 'Wake') || contains(stage_name, '|0')
    stage_label = 0;
elseif contains(stage_name, 'Stage1') || contains(stage_name, 'N1') || contains(stage_name, '|1')
    stage_label = 1;
elseif contains(stage_name, 'Stage2') || contains(stage_name, 'N2') || contains(stage_name, '|2')
    stage_label = 2;
elseif contains(stage_name, 'Stage3') || contains(stage_name, 'Stage4') || contains(stage_name, 'N3') || contains(stage_name, '|3') || contains(stage_name, '|4')
    stage_label = 3;
elseif contains(stage_name, 'REM') || contains(stage_name, '|5')
    stage_label = 4;
end

end
