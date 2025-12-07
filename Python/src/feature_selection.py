import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler
import os

# Handle both package import and standalone execution
try:
    from .utils import save_cache
except ImportError:
    from utils import save_cache

def select_features(features, labels, config):
    """
    STUDENT IMPLEMENTATION AREA: Select most relevant features.

    Feature selection becomes important in later iterations to:
    1. Reduce overfitting
    2. Improve computation speed
    3. Focus on most discriminative features
    4. Handle curse of dimensionality

    Suggested approaches for students to implement:
    - Statistical tests (ANOVA F-test, chi-square)
    - Mutual information
    - Correlation-based selection
    - Recursive feature elimination
    - L1 regularization (LASSO)
    - Tree-based feature importance

    Args:
        features (np.ndarray): The input features (n_samples, n_features).
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.

    Returns:
        np.ndarray: The selected features (n_samples, n_selected_features).
    """
    print(f"Selecting features for iteration {config.CURRENT_ITERATION}...")
    print(f"Input features shape: {features.shape}")

    if features.shape[1] == 0:
        print("⚠️  WARNING: No features to select from!")
        return features
    
    if config.CURRENT_ITERATION == 1:
        # Early iterations: Use all available features
        print("Early iteration - using all available features")
        selected_features = features

    if config.CURRENT_ITERATION == 2:
        # Normalize variance
        scaler=MinMaxScaler()
        var_scaled= scaler.fit_transform(features)
        # Variance Thresholding
        selected_var=VarianceThreshold(threshold=0.01)
        selected_var.fit(var_scaled)
        # Select features and send the original variance value to the next step
        mask_var=selected_var.get_support()
        var_selected_features=features[:,mask_var]
        
        # Save variance mask for inference
        varmask_filename = f"varmask_iter{config.CURRENT_ITERATION}.joblib"
        save_cache(mask_var, varmask_filename, config.CACHE_DIR)
        print(f"Saved variance mask to {varmask_filename}")

        #Correlation Analysis
        #Create a DataFrame
        data=pd.DataFrame(var_selected_features)

        #Correlation matrix
        corr_matrix=data.corr()

        #Select the upper bounds of the matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        #Select the columns that have a correlation bigger than a certain threshhold with any other column
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        corr_selected_features=data.drop(to_drop, axis=1)
        
        # Create correlation mask (boolean array indicating which columns to keep)
        # This mask is applied to var_selected_features (after variance thresholding)
        # to_drop contains column indices/names from the DataFrame
        # We need to create a boolean mask for the original var_selected_features columns
        n_cols_after_var = var_selected_features.shape[1]
        corrmask = np.ones(n_cols_after_var, dtype=bool)
        for col_idx, col_name in enumerate(data.columns):
            if col_name in to_drop:
                corrmask[col_idx] = False
        
        corrmask_filename = f"corrmask_iter{config.CURRENT_ITERATION}.joblib"
        save_cache(corrmask, corrmask_filename, config.CACHE_DIR)
        print(f"Saved correlation mask to {corrmask_filename}")

        #Statistical Testing (Option B:Mutual Information)
        k_features = getattr(config, 'FEATURE_SELECTION_K', 50)
        selector = SelectKBest(mutual_info_classif, k=k_features)
        MI_selected_features = selector.fit_transform(corr_selected_features, labels)
        
        # Save selector for inference
        selector_filename = f"feature_selector_iter{config.CURRENT_ITERATION}.joblib"
        save_cache(selector, selector_filename, config.CACHE_DIR)
        print(f"Saved feature selector to {selector_filename}")

        print(f"Select best {k_features} features")
        selected_features = MI_selected_features

    elif config.CURRENT_ITERATION == 3:
        # TODO: Students should implement feature selection here
        # Target: Select ~30 best features from larger set
        print("TODO: Students should implement feature selection for iteration 3")
        print("Suggested: Use SelectKBest with f_classif to select ~30 features")
        print("Example code:")
        print("  from sklearn.feature_selection import SelectKBest, f_classif")
        print("  selector = SelectKBest(f_classif, k=30)")
        print("  selected_features = selector.fit_transform(features, labels)")

        # Placeholder - students must replace:
        selected_features = features  # No selection implemented yet

    elif config.CURRENT_ITERATION == 4:
        # TODO: Students should implement advanced feature selection
        print("TODO: Students should implement advanced feature selection for iteration 4")
        print("Suggested: Use more sophisticated methods like RFE or feature importance")

        # Placeholder - students must replace:
        selected_features = features  # No selection implemented yet

    print(f"Selected features shape: {selected_features.shape}")
    return selected_features
