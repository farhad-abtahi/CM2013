function visualization_visualize_results(model, features, labels, config)
%% Visualizes the results of the classification.

fprintf('Visualizing results...\n');

% TODO: Add more visualizations as needed (e.g., feature importance).

% For simplicity, we'll just plot a confusion matrix for now.
predictions = predict(model, features);

% Convert labels to categorical for confusionchart
class_names = categorical({'Wake', 'N1', 'N2', 'N3', 'REM'});

% Ensure predictions and labels are categorical and have the same categories
predictions_cat = categorical(predictions, 0:4, class_names.categories);
labels_cat = categorical(labels, 0:4, class_names.categories);

figure;
confusionchart(labels_cat, predictions_cat);
title('Confusion Matrix');

end
