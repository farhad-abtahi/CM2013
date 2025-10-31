from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import pandas as pd
import numpy as np 

def generate_report(model, features, labels, config, processing_log):
    """
    Generates a report summarizing the results.

    For the jumpstart, this is a placeholder.

    Args:
        model (object): The trained model.
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.
    """
    print("Generating report...")
    # TODO: Implement a function to generate a comprehensive report 
    # (e.g., as a text file or PDF) that includes:
    # - Performance metrics (accuracy, kappa, F1-score)
    # - Confusion matrix
    # - Details about the model and features used
    
    # After prediction
    y_pred = model.predict(features)
    y_gt = labels

    # Calculate all metrics
    accuracy = accuracy_score(y_gt, y_pred)
    kappa = cohen_kappa_score(y_gt, y_pred)
    macro_f1 = f1_score(y_gt, y_pred, average='macro')
    weighted_f1 = f1_score(y_gt, y_pred, average='weighted')

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_gt, y_pred)

    # Detailed report
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    print(classification_report(y_gt, y_pred, target_names=stage_names))

    # Confusion matrix
    cm = confusion_matrix(y_gt, y_pred)
    cm_dataFrame = pd.DataFrame(cm, index=stage_names, columns=stage_names) # conversion of CM to data frame
    
    report_content = f"""
    {processing_log}


    # Sleep Scoring Report - Iteration {config.CURRENT_ITERATION}

    ## Model
    {type(model).__name__}

    ## Performance
    Accuracy:            {accuracy:.3f}
    Kappa:               {kappa:.3f}
    F1-score (macro):    {macro_f1:.3f}
    F1-score (weighted): {weighted_f1:.3f}

    ## Per-class Metrics
    Precision: {np.round(precision, 4)}
    Recall:    {np.round(recall, 4)}
    F1:        {np.round(f1, 4)}
    Support:   {support}

    ## Confusion Matrix
    {cm_dataFrame.to_string()}

    ## Notes
    Features shape: {features.shape}
    Full report generated.
    """
    with open("report.txt", "w") as f:
        f.write(report_content)
    print("Report saved to report.txt")
