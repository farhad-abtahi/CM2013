import numpy as np
import pandas as pd
import os

def make_inference(model, holdout_data, config):
    """
    Makes predictions on the hold-out data using the trained model.

    Args:
        model (object): The trained classification model.
        holdout_data (np.ndarray): The preprocessed and feature-extracted hold-out data.
        config (module): The configuration module.

    Returns:
        np.ndarray: Predicted labels for the hold-out data.
    """
    print("Making inference on hold-out data...")
    predictions = model.predict(holdout_data)
    return predictions

def generate_submission_file(predictions, record_numbers, epoch_numbers, config):
    """
    Generates a submission CSV file.

    Args:
        predictions (np.ndarray): The predicted sleep stage labels.
        record_numbers (list): List of record numbers corresponding to each epoch.
        epoch_numbers (list): List of epoch numbers corresponding to each epoch.
        config (module): The configuration module.
    """
    print(f"Generating submission file: {config.SUBMISSION_FILE}...")
    submission_df = pd.DataFrame({
        'record_number': record_numbers,
        'epoch_number': epoch_numbers,
        'label': predictions
    })
    submission_df.to_csv(os.path.join(config.DATA_DIR, config.SUBMISSION_FILE), index=False)
    print("Submission file generated successfully.")
