function model = train_classifier(features, labels)
%% Train classifier - wrapper that accesses config from caller's workspace
try
    CURRENT_ITERATION = evalin('caller', 'CURRENT_ITERATION');
    CLASSIFIER_TYPE = evalin('caller', 'CLASSIFIER_TYPE');
    KNN_N_NEIGHBORS = evalin('caller', 'KNN_N_NEIGHBORS');
catch
    CURRENT_ITERATION = 1;
    CLASSIFIER_TYPE = 'knn';
    KNN_N_NEIGHBORS = 5;
end

fprintf('Training %s classifier for iteration %d...\n', CLASSIFIER_TYPE, CURRENT_ITERATION);

% Simple k-NN classifier example
if strcmp(CLASSIFIER_TYPE, 'knn')
    model = fitcknn(features, labels, 'NumNeighbors', KNN_N_NEIGHBORS);
    fprintf('k-NN classifier trained with k=%d\n', KNN_N_NEIGHBORS);
else
    fprintf('TODO: Implement %s classifier\n', CLASSIFIER_TYPE);
    model = fitcknn(features, labels, 'NumNeighbors', 5); % Fallback
end

% Evaluate on training data (for demonstration)
predictions = predict(model, features);
% Ensure both are column vectors for comparison
predictions = predictions(:);
labels = labels(:);
accuracy = sum(predictions == labels) / length(labels) * 100;
fprintf('Training accuracy: %.2f%%\n', accuracy);
end
