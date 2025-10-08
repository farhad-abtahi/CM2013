function test_pipeline()
%% Basic pipeline integration tests for MATLAB implementation.
%% These test the overall flow and can help students debug issues.

fprintf('Running MATLAB pipeline tests...\n');

% Add parent directory to path
addpath('..');
addpath('../src');

% Test 1: Main script can be called without errors (structure test)
try
    % We don't actually run main() since functions might not be implemented
    % Just check that the file exists and can be parsed
    if ~exist('../main.m', 'file')
        error('main.m file not found');
    end
    fprintf('✓ Main script file exists\n');
catch ME
    error('✗ Main script structure test failed: %s', ME.message);
end

% Test 2: Required function files exist in src/
required_functions = {'data_loader.m', 'preprocessing.m', 'feature_extraction.m', ...
                     'feature_selection.m', 'classification.m', 'visualization.m', ...
                     'report.m', 'inference.m', 'utils.m'};

missing_functions = {};
for i = 1:length(required_functions)
    func_file = fullfile('../src', required_functions{i});
    if ~exist(func_file, 'file')
        missing_functions{end+1} = required_functions{i}; %#ok<AGROW>
    end
end

if ~isempty(missing_functions)
    warning('✗ Missing function files: %s', strjoin(missing_functions, ', '));
    fprintf('  Note: Students need to create these function files\n');
else
    fprintf('✓ All required function files exist\n');
end

% Test 3: Cache utilities exist
utils_dir = '../src/utils';
if exist(utils_dir, 'dir')
    if exist(fullfile(utils_dir, 'save_cache.m'), 'file') && ...
       exist(fullfile(utils_dir, 'load_cache.m'), 'file')
        fprintf('✓ Cache utility functions exist\n');
    else
        fprintf('! Cache utility functions missing - students need to implement\n');
    end
else
    fprintf('! Utils directory missing - students need to create\n');
end

% Test 4: Configuration integration
try
    run('../config.m');
    fprintf('✓ Configuration integrates properly with pipeline structure\n');
catch ME
    error('✗ Configuration integration failed: %s', ME.message);
end

fprintf('\nPipeline structure tests completed!\n');
fprintf('Note: Function implementation tests will depend on student code.\n');

end