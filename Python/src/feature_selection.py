import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler

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

        #Statistical Testing (Option B:Mutual Information)
        MI_selected_features=SelectKBest(mutual_info_classif,k=40).fit_transform(corr_selected_features, labels)

        print("Select best 40 features")
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
