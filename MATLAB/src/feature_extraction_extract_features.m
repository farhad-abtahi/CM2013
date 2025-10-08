function features = feature_extraction_extract_features(data, CURRENT_ITERATION)
%% STUDENT IMPLEMENTATION AREA: Extract features based on current iteration.

% EXAMPLE: Very basic features (students should expand significantly!)
n_epochs = size(data, 1);
n_features_per_epoch = 3; % mean, median, std (students should add 13+ more!)

features = zeros(n_epochs, n_features_per_epoch);
for i = 1:n_epochs
    epoch = data(i, :);
    features(i, 1) = mean(epoch);
    features(i, 2) = median(epoch);
    features(i, 3) = std(epoch);
end

fprintf('Extracted %d features from %d epochs\n', n_features_per_epoch, n_epochs);
fprintf('WARNING: Only 3 basic features implemented. Students must add 13+ more!\n');
end
