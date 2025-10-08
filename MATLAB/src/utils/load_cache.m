function data = load_cache(filename, cache_dir)
%% Loads data from a cache file.

filepath = fullfile(cache_dir, filename);
if exist(filepath, 'file')
    fprintf('Loading data from cache: %s\n', filepath);
    loaded_data = load(filepath);
    data = loaded_data.data;
else
    fprintf('Cache file not found: %s\n', filepath);
    data = []; % Return empty if not found
end

end
