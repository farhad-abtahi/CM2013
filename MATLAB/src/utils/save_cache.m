function save_cache(data, filename, cache_dir)
%% Saves data to a cache file.

if ~exist(cache_dir, 'dir')
    mkdir(cache_dir);
end

filepath = fullfile(cache_dir, filename);
save(filepath, 'data');
fprintf('Data cached to %s\n', filepath);

end
