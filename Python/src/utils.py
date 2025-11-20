import os
import numpy as np
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

def add_contextual_features(features, n_prev=2, n_next=2):
    #Concatenate features from previous and next n epochs to the current epoch.

    n_epochs, n_feats = features.shape
    features_context = []
    
    # Handle boundary conditions using Edge Padding
    padded_features = np.pad(features, ((n_prev, n_next), (0, 0)), mode='edge')

    for i in range(n_epochs):
        # Define the window range in the padded array: [i : i + n_prev + 1 + n_next]
        window = padded_features[i : i + n_prev + n_next + 1, :]
        # Flatten the window into a single 1D vector
        features_context.append(window.flatten())
        
    return np.array(features_context)