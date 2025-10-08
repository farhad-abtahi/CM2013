import os
import joblib

def save_cache(data, filename, cache_dir):
    """
    Saves data to a cache file.

    Args:
        data (any): The data to be cached.
        filename (str): The name of the cache file.
        cache_dir (str): The directory to save the cache file.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filepath = os.path.join(cache_dir, filename)
    joblib.dump(data, filepath)
    print(f"Data cached to {filepath}")

def load_cache(filename, cache_dir):
    """
    Loads data from a cache file.

    Args:
        filename (str): The name of the cache file.
        cache_dir (str): The directory where the cache file is located.

    Returns:
        any: The loaded data, or None if the file does not exist.
    """
    filepath = os.path.join(cache_dir, filename)
    if os.path.exists(filepath):
        print(f"Loading data from cache: {filepath}")
        return joblib.load(filepath)
    print(f"Cache file not found: {filepath}")
    return None
