function selected_features = feature_selection_select_features(features, labels)
%% Selects the most relevant features.
% For the jumpstart, this is a placeholder.

fprintf('Selecting features...\n');

if config.CURRENT_ITERATION >= 3
    % TODO: Implement feature selection (e.g., using MATLAB's `fscchi2` or similar)
    fprintf('Warning: No feature selection defined for this iteration. Returning all features.\n');
    selected_features = features;
else
    % No feature selection needed for early iterations
    selected_features = features;
end

end
