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

def calculate_sleep_metrics(labels, epoch_duration=30):
    """
    Calculate sleep architecture metrics from epoch labels.

    Args:
        labels: array of sleep stage labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        epoch_duration: seconds per epoch (default 30)

    Returns:
        metrics: dict of sleep architecture values
    """

    labels = np.array(labels)
    recording_time_min = (len(labels) * epoch_duration) / 60.0

    # Students must implement based on definitions above
    metrics = {
        "Total recording time [min]": recording_time_min
    }

    # 1. Find sleep onset (first non-wake epoch)

    sleep_onset = np.where(labels != 0)[0]

    # 2. Calculate SOL, REM latency, TST, WASO

    # SOL
    if len(sleep_onset) > 0:
        first_sleep = sleep_onset[0]
        sol_min = (first_sleep * epoch_duration) / 60.0
    else:
        first_sleep = None
        sol_min = np.nan
    metrics["Sleep Onset Latenancy (SOL) [min]"] = sol_min

    # REM latenancy
    rem_epochs = np.where(labels == 4)[0]
    if len(rem_epochs) > 0 and first_sleep is not None:
        first_rem = rem_epochs[0]
        rem_lat_min = (first_rem - first_sleep) * epoch_duration / 60.0
    else:
        first_rem = None
        rem_lat_min = np.nan
    metrics["REM Latency [min]"] = rem_lat_min

    # TST
    tst_min = np.sum(labels != 0) * epoch_duration / 60.0
    tst_hr = tst_min / 60.0
    metrics["Total Sleep Time (TST) [hr]"] = tst_hr

    # WASO
    if first_sleep is not None:
        waso_total = (labels[first_sleep:] == 0)
        waso_min = np.sum(waso_total) * epoch_duration / 60.0
    else:
        waso_min = np.nan
    metrics["Wake After Sleep Onset (WASO) [min]"] = waso_min


    # 3. Calculate stage percentages
    stages = [0,1,2,3,4]
    stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    for i, stage in enumerate(stages):
        stage_length = np.sum(labels == i) * epoch_duration / 60.0
        if tst_min > 0:
            stage_percentage = (stage_length / tst_min * 100.0)
        else:
            stage_percentage = np.nan
        metrics[f"{stage_names[i]} [%]"] = stage_percentage
              

    # 4. Count awakenings
    awakenings = 0
    for i in range(1, len(labels)):
        if labels[i] == 0 and labels[i-1] != 0:
            awakenings += 1
    metrics["Number of awakenings"] = awakenings

    # 5. Sleep Efficiency (SE)
    if recording_time_min > 0:
        se = tst_min / recording_time_min * 100.0
    else:
        se = np.nan
    metrics["Sleep Efficiency (SE) [%]"] = se

    # REM Cycle Count and Duration
    rem_parts = []
    in_rem = False
    start_point = None
    end_point = None

    for i, stage in enumerate(labels):
        if stage == 4 and not in_rem:
            in_rem = True
            start_point = i
        elif stage != 4 and in_rem:
            in_rem = False
            end_point = i - 1
            rem_parts.append((start_point, end_point))
    if in_rem and start_point is not None:
        end_point = len(labels) - 1
        rem_parts.append((start_point, end_point))
    metrics["REM cycle"] = len(rem_parts)

    if len(rem_parts) > 0:
        rem_dur = []
        for part in rem_parts:
            part_start, part_stop = part
            dur_min = (part_stop - part_start + 1) * epoch_duration / 60.0
            rem_dur.append(dur_min)
        metrics["Mean REM duration"] = np.mean(rem_dur)
    else:
        metrics["Mean REM duration"] = np.nan

    return metrics


def generate_report(model, features, labels, config, processing_log, y_true_all=None, y_pred_all=None):
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

    y_gt = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    # Compare ground truth vs predictions
    true_metrics = calculate_sleep_metrics(y_gt)
    pred_metrics = calculate_sleep_metrics(y_pred)

    #Space in the report.txt
    clinical_section = "\n## Clinical Validation\n"
    clinical_section += f"{'Metric':35s} {'True':>10s} {'Pred':>10s} {'Error':>10s}\n"
    clinical_section += "-" * 70 + "\n"

    # Report differences
    for metric_name in true_metrics:
        true_val = true_metrics[metric_name]
        pred_val = pred_metrics[metric_name]
        error = abs(pred_val - true_val)
        clinical_section += f"{metric_name:35s} {true_val:10.1f} {pred_val:10.1f} {error:10.1f}\n"
        #print(f"{metric_name}: True={true_val:.1f}, Pred={pred_val:.1f}, Error={error:.1f}")

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

    {clinical_section}

    ## Notes
    Features shape: {features.shape}
    Full report generated.
    """
    with open("report.txt", "w") as f:
        f.write(report_content)
    print("Report saved to report.txt")
