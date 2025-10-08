function generate_report(model, features, labels)
%% Generate report - simple text report
fprintf('\n=== SLEEP SCORING REPORT ===\n');

% Predict
predictions = predict(model, features);

% Calculate metrics
accuracy = sum(predictions == labels) / length(labels) * 100;
fprintf('Overall Accuracy: %.2f%%\n', accuracy);

% Per-class accuracy
stage_names = {'Wake', 'N1', 'N2', 'N3', 'REM'};
for stage = 0:4
    idx = labels == stage;
    if sum(idx) > 0
        stage_acc = sum(predictions(idx) == labels(idx)) / sum(idx) * 100;
        fprintf('%s Accuracy: %.2f%% (%d epochs)\n', stage_names{stage+1}, stage_acc, sum(idx));
    end
end

fprintf('\nReport generation complete\n');
end
