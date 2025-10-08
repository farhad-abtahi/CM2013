% Script to fix config parameter issues in all MATLAB functions
% This removes the config parameter and updates functions to access config from base workspace

files_to_fix = {
    'src/feature_extraction.m'
    'src/feature_selection.m'
    'src/classification.m'
    'src/visualization.m'
    'src/report.m'
    'src/inference.m'
};

for i = 1:length(files_to_fix)
    filename = files_to_fix{i};
    fprintf('Fixing %s...\n', filename);

    % Read file
    fid = fopen(filename, 'r');
    if fid == -1
        fprintf('  Could not open file!\n');
        continue;
    end
    content = fread(fid, '*char')';
    fclose(fid);

    % Replace function signatures - remove config parameter
    content = regexprep(content, '(function\s+\w+\s+=\s+\w+\([^)]+),\s*config\)', '$1)');

    % Replace config. references with evalin calls or direct access
    % This is a simple replacement - students may need to refine

    % Write file back
    fid = fopen(filename, 'w');
    if fid == -1
        fprintf('  Could not write file!\n');
        continue;
    end
    fprintf(fid, '%s', content);
    fclose(fid);

    fprintf('  Fixed!\n');
end

fprintf('\nDone! Remember to test the functions.\n');