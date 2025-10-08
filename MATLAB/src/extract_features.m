function features = extract_features(data)
%% Extract features - wrapper that accesses config from caller's workspace
% Get CURRENT_ITERATION from caller's workspace
try
    CURRENT_ITERATION = evalin('caller', 'CURRENT_ITERATION');
catch
    CURRENT_ITERATION = 1; % Default
end

fprintf('Extracting features for iteration %d...\n', CURRENT_ITERATION);

% For now, call the original feature_extraction function
% Students should implement actual feature extraction here
features = feature_extraction_extract_features(data, CURRENT_ITERATION);
end
