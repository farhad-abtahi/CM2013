function model = train_classifier(features, labels)
%% STUDENT IMPLEMENTATION AREA: Train classifier based on iteration.
%
% This function provides a basic framework but students should enhance it:
%
% 1. Implement proper cross-validation (not just train/test split)
% 2. Address class imbalance in sleep stage data
% 3. Tune hyperparameters for each classifier
% 4. Add more sophisticated evaluation metrics
% 5. Consider ensemble methods in later iterations

% Get config variables from caller's workspace
try
    CURRENT_ITERATION = evalin('caller', 'CURRENT_ITERATION');
    CLASSIFIER_TYPE = evalin('caller', 'CLASSIFIER_TYPE');
    KNN_N_NEIGHBORS = evalin('caller', 'KNN_N_NEIGHBORS');
catch
    CURRENT_ITERATION = 1;
    CLASSIFIER_TYPE = 'knn';
    KNN_N_NEIGHBORS = 5;
end

fprintf('Training %s classifier...\n', CLASSIFIER_TYPE);
fprintf('Features shape: [%d, %d], Labels length: %d\n', size(features), length(labels));

% Basic validation
if size(features, 1) == 0 || size(features, 2) == 0
    error('No features available for training!');
end

% BASIC train/test split - students should implement cross-validation
% TODO: Students should implement k-fold cross-validation for more robust evaluation
% Use stratified split for realistic sleep data distribution
% Sleep stages are naturally imbalanced (more N2, less N1/REM)
cv = cvpartition(labels, 'HoldOut', 0.2, 'Stratify', true);
X_train = features(training(cv), :);
X_test = features(test(cv), :);
y_train = labels(training(cv));
y_test = labels(test(cv));

fprintf('Using stratified train/test split to maintain class balance\n');
fprintf('Training set: %d samples, Test set: %d samples\n', size(X_train, 1), size(X_test, 1));

% TODO: Students should address class imbalance in sleep data:
% - Sleep stages are not equally distributed
% - Consider SMOTE, class weights, or other techniques

% Select classifier based on iteration (using config parameters)
if CURRENT_ITERATION == 1
    % Iteration 1: Simple k-NN
    model = fitcknn(X_train, y_train, 'NumNeighbors', KNN_N_NEIGHBORS);
    fprintf('Using k-NN with k=%d\n', KNN_N_NEIGHBORS);

elseif CURRENT_ITERATION == 2
    % Iteration 2: SVM
    % TODO: Students should tune hyperparameters
    try
        SVM_KERNEL_SCALE = evalin('caller', 'SVM_KERNEL_SCALE');
        model = fitcsvm(X_train, y_train, 'KernelScale', SVM_KERNEL_SCALE);
    catch
        model = fitcsvm(X_train, y_train);
    end
    fprintf('Using SVM\n');

elseif CURRENT_ITERATION >= 3
    % Iteration 3+: Random Forest
    % TODO: Students should tune hyperparameters
    try
        RF_N_TREES = evalin('caller', 'RF_N_TREES');
        n_trees = RF_N_TREES;
    catch
        n_trees = 100;
    end
    model = TreeBagger(n_trees, X_train, y_train, 'Method', 'classification');
    fprintf('Using Random Forest with %d trees\n', n_trees);

else
    error('Invalid iteration: %d', CURRENT_ITERATION);
end

% Train the model (already done in fitc* functions for MATLAB)
fprintf('Training model...\n');

% Comprehensive evaluation with detailed performance metrics
if CURRENT_ITERATION >= 3 && isa(model, 'TreeBagger')
    y_pred_cell = predict(model, X_test);
    y_pred = str2double(y_pred_cell); % Convert cell to numeric
else
    y_pred = predict(model, X_test);
end

overall_accuracy = sum(y_pred == y_test) / length(y_test);
fprintf('Overall accuracy: %.3f\n', overall_accuracy);

% Calculate and display detailed performance metrics
print_performance_metrics(y_test, y_pred);

% TODO: Students should add more advanced metrics:
% - Cohen's kappa (important for sleep scoring)
% - ROC-AUC for each class
% - Cross-validation scores
% - Feature importance analysis
fprintf('\nTODO: Students should add Cohen''s kappa and ROC-AUC metrics\n');

end


function print_performance_metrics(y_true, y_pred)
%% Print comprehensive performance metrics for sleep stage classification.
%
% Includes accuracy, sensitivity (recall), specificity, and F1-score for each sleep stage.

% Sleep stage labels and names (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
stage_names = {'Wake', 'N1', 'N2', 'N3', 'REM'};
stage_labels = 0:4;

fprintf('\n%s\n', repmat('=', 1, 70));
fprintf('SLEEP STAGE CLASSIFICATION PERFORMANCE METRICS\n');
fprintf('%s\n', repmat('=', 1, 70));

% Overall metrics
overall_accuracy = sum(y_pred == y_true) / length(y_true);
fprintf('Overall Accuracy: %.3f\n', overall_accuracy);

% Confusion Matrix
fprintf('\nConfusion Matrix:\n');
C = confusionmat(y_true, y_pred, 'Order', stage_labels);

% Display confusion matrix with labels
fprintf('       ');
for j = 1:length(stage_names)
    fprintf('%6s', stage_names{j});
end
fprintf('\n');

for i = 1:length(stage_names)
    fprintf('%6s ', stage_names{i});
    for j = 1:length(stage_names)
        fprintf('%6d', C(i, j));
    end
    fprintf('\n');
end

% Per-class metrics
fprintf('\nPer-Class Performance Metrics:\n');
fprintf('%s\n', repmat('-', 1, 70));
fprintf('%-8s %-10s %-12s %-12s %-10s\n', 'Stage', 'Accuracy', 'Sensitivity', 'Specificity', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 70));

% Calculate metrics for each sleep stage
for i = 1:length(stage_names)
    stage_idx = stage_labels(i);
    stage_name = stage_names{i};

    if any(y_true == stage_idx) % Only calculate if stage is present in test set
        % Per-class accuracy (percentage of this class correctly classified)
        class_mask = (y_true == stage_idx);
        if sum(class_mask) > 0
            class_accuracy = sum((y_pred == stage_idx) & (y_true == stage_idx)) / sum(class_mask);
        else
            class_accuracy = 0.0;
        end

        % Sensitivity (Recall) - True Positive Rate
        tp = sum((y_true == stage_idx) & (y_pred == stage_idx));
        fn = sum((y_true == stage_idx) & (y_pred ~= stage_idx));
        sensitivity = tp / (tp + fn);
        if isnan(sensitivity), sensitivity = 0; end

        % Specificity - True Negative Rate
        tn = sum((y_true ~= stage_idx) & (y_pred ~= stage_idx));
        fp = sum((y_true ~= stage_idx) & (y_pred == stage_idx));
        specificity = tn / (tn + fp);
        if isnan(specificity), specificity = 0; end

        % F1-Score
        precision = tp / (tp + fp);
        if isnan(precision), precision = 0; end
        if precision + sensitivity > 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity);
        else
            f1 = 0;
        end

        fprintf('%-8s %-10.3f %-12.3f %-12.3f %-10.3f\n', ...
                stage_name, class_accuracy, sensitivity, specificity, f1);
    else
        fprintf('%-8s %-10s %-12s %-12s %-10s\n', ...
                stage_name, 'N/A', 'N/A', 'N/A', 'N/A');
    end
end

fprintf('%s\n', repmat('-', 1, 70));

% Class distribution in test set
fprintf('\nClass Distribution in Test Set:\n');
for i = 1:length(stage_names)
    stage_idx = stage_labels(i);
    stage_name = stage_names{i};
    count = sum(y_true == stage_idx);
    percentage = count / length(y_true) * 100;
    fprintf('%s: %d samples (%.1f%%)\n', stage_name, count, percentage);
end

% Sleep scoring specific notes
fprintf('\nNotes for Sleep Scoring:\n');
fprintf('- Sensitivity = Recall = True Positive Rate (correctly identified stages)\n');
fprintf('- Specificity = True Negative Rate (correctly rejected stages)\n');
fprintf('- Sleep stage imbalance is natural (more N2, less N1/REM)\n');
fprintf('- Consider Cohen''s kappa for chance-corrected agreement\n');
fprintf('- Clinical focus: High sensitivity for REM and N3 stages\n');

end
