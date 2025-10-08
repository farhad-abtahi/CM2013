function visualize_results(model, features, labels)
%% Visualize results - simple confusion matrix
fprintf('Generating visualization...\n');

% Predict on training data
predictions = predict(model, features);

% Ensure both labels and predictions are column vectors
labels = labels(:);
predictions = predictions(:);

% Sleep stage label names (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
stage_names = {'Wake', 'N1', 'N2', 'N3', 'REM'};

% Convert numeric labels to categorical with proper names
labels_cat = categorical(labels, 0:4, stage_names);
predictions_cat = categorical(predictions, 0:4, stage_names);

% Create confusion matrix with proper labels
figure;
cm = confusionchart(labels_cat, predictions_cat);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
cm.XLabel = 'Predicted Sleep Stage';
cm.YLabel = 'True Sleep Stage';
cm.Title = 'Sleep Scoring Confusion Matrix';

fprintf('Visualization complete\n');
end
