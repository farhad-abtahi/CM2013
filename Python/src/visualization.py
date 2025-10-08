import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xml.etree.ElementTree as ET

# Try to import MNE for EDF reading (more lenient than pyedflib)
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False
    try:
        import pyedflib
        HAS_PYEDFLIB = True
    except ImportError:
        HAS_PYEDFLIB = False

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plots a confusion matrix.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        class_names (list): The names of the classes.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

def plot_sample_epoch(edf_path, epoch_idx=0, epoch_duration=30):
    """
    Plot all signals from a sample epoch in an EDF file.

    Args:
        edf_path (str): Path to the EDF file.
        epoch_idx (int): Index of the epoch to plot (default: 0).
        epoch_duration (int): Duration of each epoch in seconds (default: 30).
    """
    if not HAS_MNE and not HAS_PYEDFLIB:
        print("Error: Neither MNE nor pyedflib is installed.")
        print("Please install one: pip install mne  OR  pip install pyedflib")
        return

    try:
        # Reset matplotlib to defaults
        import matplotlib
        matplotlib.rcdefaults()

        # Calculate epoch boundaries
        start_time = epoch_idx * epoch_duration

        if HAS_MNE:
            # Use MNE (more lenient with EDF format issues)
            raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)

            n_channels = len(raw.ch_names)
            channel_labels = raw.ch_names

            # Extract data for this epoch
            start_sample = int(start_time * raw.info['sfreq'])
            stop_sample = int((start_time + epoch_duration) * raw.info['sfreq'])

            # Convert from Volts to microvolts for better visualization
            # MNE loads data in Volts by default
            data_all = raw[:, start_sample:stop_sample][0] * 1e6
            times = np.arange(data_all.shape[1]) / raw.info['sfreq'] + start_time

        else:
            # Fallback to pyedflib
            with pyedflib.EdfReader(edf_path) as edf:
                n_channels = edf.signals_in_file
                channel_labels = edf.getSignalLabels()
                sampling_freqs = [edf.getSampleFrequency(i) for i in range(n_channels)]

                data_all = []
                for ch_idx in range(n_channels):
                    fs = sampling_freqs[ch_idx]
                    start_sample = int(start_time * fs)
                    n_samples = int(epoch_duration * fs)
                    signal = edf.readSignal(ch_idx, start=start_sample, n=n_samples)
                    data_all.append(signal)

                # Create time axis
                max_samples = max(len(d) for d in data_all)
                times = np.linspace(start_time, start_time + epoch_duration, max_samples)

        # Create subplots - EXACTLY like the diagnostic plot that worked
        fig, axes = plt.subplots(n_channels, 1, figsize=(14, 2*n_channels),
                                facecolor='white', edgecolor='black')
        if n_channels == 1:
            axes = [axes]

        print(f"\nPlotting Epoch {epoch_idx} (Time: {start_time}-{start_time+epoch_duration}s)")
        print("="*70)

        for ch_idx in range(n_channels):
            label = channel_labels[ch_idx]

            if HAS_MNE:
                signal = data_all[ch_idx]
            else:
                signal = data_all[ch_idx]

            # Set white background for subplot - EXACTLY like diagnostic
            axes[ch_idx].set_facecolor('white')

            # Plot with VERY visible settings - EXACTLY like diagnostic
            axes[ch_idx].plot(times, signal, 'b-', linewidth=2.0, solid_capstyle='round')

            # Add unit to ylabel for bio-signal channels
            if 'EEG' in label or 'EOG' in label or 'EMG' in label or 'ECG' in label:
                ylabel = f'{label} (µV)'
            else:
                ylabel = f'{label}'

            axes[ch_idx].set_ylabel(ylabel, fontsize=11, fontweight='bold')
            axes[ch_idx].grid(True, color='gray', alpha=0.4, linestyle='-', linewidth=0.5)
            axes[ch_idx].set_xlim(times[0], times[-1])

            # Explicit y-limits - EXACTLY like diagnostic
            y_margin = (signal.max() - signal.min()) * 0.15
            axes[ch_idx].set_ylim(signal.min() - y_margin, signal.max() + y_margin)

            # Add text showing we have data - EXACTLY like diagnostic
            axes[ch_idx].text(0.98, 0.95, f'n={len(signal)}', transform=axes[ch_idx].transAxes,
                            ha='right', va='top', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

            print(f"  {label}: {len(signal)} samples, range=[{signal.min():.1f}, {signal.max():.1f}]")

        axes[-1].set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Sleep Signals - Epoch {epoch_idx} ({epoch_duration}s window)',
                         fontsize=14, fontweight='bold')

        plt.tight_layout()

        # Save figure explicitly before showing
        output_path = f"epoch{epoch_idx}_signals.png"
        plt.savefig(output_path, dpi=100, facecolor='white', edgecolor='black', bbox_inches='tight')
        print(f"\n✓ Saved to {output_path}")

        plt.show()

    except FileNotFoundError:
        print(f"Error: EDF file not found at {edf_path}")
    except Exception as e:
        print(f"Error reading EDF file: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_hypnogram(xml_path, edf_path=None):
    """
    Plot hypnogram (sleep stage progression) from XML annotations.

    Args:
        xml_path (str): Path to the XML annotation file.
        edf_path (str, optional): Path to EDF file to get recording duration.
    """
    try:
        # Parse XML file
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Extract sleep stages and times
        epochs = []
        stages = []

        # Try different XML structures (Compumedics format)
        for event in root.findall('.//ScoredEvent'):
            event_concept = event.find('EventConcept')
            start = event.find('Start')
            duration = event.find('Duration')

            if event_concept is not None and start is not None:
                stage_name = event_concept.text

                # Check if this is a sleep stage event
                # Formats: SDO:WakeState, SDO:NonRapidEyeMovementSleep-N1, SDO:RapidEyeMovementSleep
                # Also support older formats: Wake|0, Stage1|1, etc.
                # Exclude arousal events and other non-stage events
                is_sleep_stage = False
                if 'WakeState' in stage_name or 'RapidEyeMovementSleep' in stage_name or 'NonRapidEyeMovementSleep' in stage_name:
                    is_sleep_stage = True
                elif 'Wake|' in stage_name or 'REM|' in stage_name:
                    is_sleep_stage = True
                elif any(f'Stage{i}' in stage_name for i in range(1, 5)):
                    is_sleep_stage = True
                elif any(f'|{i}' in stage_name for i in range(6)):
                    is_sleep_stage = True

                if is_sleep_stage:
                    start_time = float(start.text)
                    dur = float(duration.text) if duration is not None else 30.0

                    # Map stage names to numeric labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
                    stage_label = None

                    if 'WakeState' in stage_name or stage_name == 'Wake' or 'Wake|0' in stage_name:
                        stage_label = 0
                    elif 'N1' in stage_name or 'Stage1' in stage_name or '|1' in stage_name:
                        stage_label = 1
                    elif 'N2' in stage_name or 'Stage2' in stage_name or '|2' in stage_name:
                        stage_label = 2
                    elif 'N3' in stage_name or 'Stage3' in stage_name or 'Stage4' in stage_name or '|3' in stage_name or '|4' in stage_name:
                        stage_label = 3
                    elif 'RapidEyeMovementSleep' in stage_name or stage_name == 'REM' or '|5' in stage_name:
                        stage_label = 4

                    if stage_label is not None:
                        # Store start time, duration, and stage label
                        epochs.append((start_time, dur, stage_label))

        if not epochs:
            print("Warning: No sleep stage annotations found in XML file")
            print("The XML file may be in a different format or empty")
            return

        # Sort epochs by start time
        epochs = sorted(epochs, key=lambda x: x[0])

        # Create hypnogram plot
        fig, ax = plt.subplots(figsize=(15, 5))

        # Plot as step function
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        stage_colors = ['red', 'orange', 'green', 'blue', 'purple']

        # Create step plot - each event has (start_time, duration, stage_label)
        for start_time, duration, stage_label in epochs:
            start_epoch = start_time / 30.0
            end_epoch = (start_time + duration) / 30.0
            ax.hlines(stage_label, start_epoch, end_epoch,
                     colors=stage_colors[int(stage_label)], linewidth=2)

        # Extract stages for statistics
        stages = np.array([e[2] for e in epochs])
        total_duration = sum(e[1] for e in epochs)

        # Styling
        ax.set_yticks(range(5))
        ax.set_yticklabels(stage_names)
        ax.set_ylabel('Sleep Stage', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch Number (30s epochs)', fontsize=12, fontweight='bold')
        ax.set_title('Hypnogram - Sleep Stage Progression', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_ylim(-0.5, 4.5)

        # Add time axis on top
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        # Convert epochs to hours
        max_epoch = (epochs[-1][0] + epochs[-1][1]) / 30.0  # Last event end time
        hour_ticks = np.arange(0, max_epoch, 120)  # 120 epochs = 1 hour
        ax2.set_xticks(hour_ticks)
        ax2.set_xticklabels([f'{h/120:.1f}' for h in hour_ticks])

        plt.tight_layout()
        plt.show()

        # Print statistics
        print("\nSleep Stage Statistics:")
        print("="*70)
        print(f"Total sleep stage events: {len(stages)}")
        print(f"Total duration: {total_duration/3600:.2f} hours")
        print(f"Total epochs: {int(total_duration/30)}")
        print("\nStage distribution:")
        for stage_idx, stage_name in enumerate(stage_names):
            # Count events and calculate total duration for this stage
            stage_events = [e for e in epochs if e[2] == stage_idx]
            count = len(stage_events)
            stage_duration = sum(e[1] for e in stage_events)
            percentage = stage_duration / total_duration * 100
            n_epochs = int(stage_duration / 30)
            print(f"  {stage_name}: {count} events, {n_epochs} epochs ({percentage:.1f}%)")

    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_path}")
    except Exception as e:
        print(f"Error reading XML file: {str(e)}")
        import traceback
        traceback.print_exc()

def visualize_results(model, features, labels, config):
    """
    Visualizes the results of the classification.

    Args:
        model (object): The trained model.
        features (np.ndarray): The input features.
        labels (np.ndarray): The corresponding labels.
        config (module): The configuration module.
    """
    print("Visualizing results...")
    # TODO: Add more visualizations as needed (e.g., feature importance).
    class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
    y_pred = model.predict(features)
    plot_confusion_matrix(labels, y_pred, class_names)
