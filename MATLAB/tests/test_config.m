function test_config()
%% Basic tests to validate MATLAB configuration and setup.
%% Students can use these as examples for more comprehensive testing.

fprintf('Running MATLAB configuration tests...\n');

% Add parent directory to path to access config
addpath('..');

% Test 1: Configuration loads without error
try
    run('../config.m');
    fprintf('✓ Configuration loaded successfully\n');
catch ME
    error('✗ Failed to load configuration: %s', ME.message);
end

% Test 2: Essential variables exist
required_vars = {'CURRENT_ITERATION', 'USE_CACHE', 'DATA_DIR', 'SAMPLE_DIR', ...
                'CACHE_DIR', 'CLASSIFIER_TYPE'};
for i = 1:length(required_vars)
    var_name = required_vars{i};
    if ~exist(var_name, 'var')
        error('✗ Required variable not found: %s', var_name);
    end
end
fprintf('✓ All required configuration variables exist\n');

% Test 3: Data directories exist
if ~exist(DATA_DIR, 'dir')
    error('✗ Data directory not found: %s', DATA_DIR);
end
if ~exist(SAMPLE_DIR, 'dir')
    error('✗ Sample directory not found: %s', SAMPLE_DIR);
end
fprintf('✓ Data directories exist\n');

% Test 4: Sample files exist
dummy_edf = fullfile(SAMPLE_DIR, 'dummy.edf');
dummy_xml = fullfile(SAMPLE_DIR, 'dummy.xml');
if ~exist(dummy_edf, 'file')
    error('✗ Sample EDF file not found: %s', dummy_edf);
end
if ~exist(dummy_xml, 'file')
    error('✗ Sample XML file not found: %s', dummy_xml);
end
fprintf('✓ Sample files exist\n');

% Test 5: Iteration configuration is valid
if CURRENT_ITERATION < 1 || CURRENT_ITERATION > 4
    error('✗ Invalid CURRENT_ITERATION: %d (must be 1-4)', CURRENT_ITERATION);
end
valid_classifiers = {'knn', 'svm', 'random_forest'};
if ~any(strcmp(CLASSIFIER_TYPE, valid_classifiers))
    error('✗ Invalid CLASSIFIER_TYPE: %s', CLASSIFIER_TYPE);
end
fprintf('✓ Iteration configuration is valid\n');

% Test 6: Cache directory was created
if ~exist(CACHE_DIR, 'dir')
    error('✗ Cache directory not created: %s', CACHE_DIR);
end
fprintf('✓ Cache directory exists\n');

fprintf('\nAll configuration tests passed! ✓\n');

end