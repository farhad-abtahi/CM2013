# Visualization Guide

This guide explains how to visualize EDF signals and hypnograms from XML annotations.

## Python Visualization

### Plot Signals from EDF File

The `plot_sample_epoch()` function reads an EDF file and plots all available signals (EEG, EOG, EMG, ECG, etc.) for a specific 30-second epoch with correct channel labels and sampling frequencies.

**Usage:**

```python
from src.visualization import plot_sample_epoch

# Plot first epoch (epoch 0)
plot_sample_epoch('data/training/recording1.edf', epoch_idx=0)

# Plot epoch 100
plot_sample_epoch('data/training/recording1.edf', epoch_idx=100)

# Plot with custom epoch duration
plot_sample_epoch('data/training/recording1.edf', epoch_idx=50, epoch_duration=30)
```

**Using the command-line script:**

```bash
cd Python

# Plot signals from a specific epoch
python plot_signals.py --edf data/training/recording1.edf --epoch 100

# Plot both signals and hypnogram
python plot_signals.py --edf data/training/recording1.edf --xml data/training/recording1.xml --epoch 50

# Plot only hypnogram
python plot_signals.py --xml data/training/recording1.xml
```

### Plot Hypnogram from XML Annotations

The `plot_hypnogram()` function reads XML annotation files and creates a hypnogram showing sleep stage progression over time.

**Usage:**

```python
from src.visualization import plot_hypnogram

# Plot hypnogram
plot_hypnogram('data/training/recording1.xml')

# Plot hypnogram with EDF file reference
plot_hypnogram('data/training/recording1.xml', edf_path='data/training/recording1.edf')
```

**Features:**
- Color-coded sleep stages: Wake (red), N1 (orange), N2 (green), N3 (blue), REM (purple)
- Dual x-axis: Epoch numbers and time in hours
- Sleep stage statistics (counts and percentages)
- Handles Compumedics XML format

## MATLAB Visualization

### Plot Signals from EDF File

**Usage:**

```matlab
cd MATLAB

% Plot first epoch (epoch 0)
plot_sample_epoch('data/training/recording1.edf', 0)

% Plot epoch 100
plot_sample_epoch('data/training/recording1.edf', 100)

% Plot with custom epoch duration
plot_sample_epoch('data/training/recording1.edf', 50, 30)
```

**Note:** MATLAB requires an EDF reader. Options:
- **Biosig Toolbox**: Download from http://biosig.sourceforge.net/
- **EEGLAB**: Contains EDF reading capabilities
- **Custom reader**: The jumpstart includes a stub `read_edf.m` that can be expanded

### Plot Hypnogram from XML Annotations

**Usage:**

```matlab
cd MATLAB

% Plot hypnogram
plot_hypnogram('data/training/recording1.xml')
```

**Features:**
- Color-coded sleep stages
- Dual x-axis: Epoch numbers and time in hours
- Sleep stage statistics printed to console
- Handles Compumedics XML format

## Signal Information

The visualizations automatically detect and display the correct sampling frequencies for each channel:

| Signal Type | Channels | Sampling Rate | Description |
|------------|----------|---------------|-------------|
| EEG | C3-A2, C4-A1 | 125 Hz | Brain activity |
| EOG | EOG(L), EOG(R) | 50 Hz | Eye movements |
| EMG | Chin EMG | 125 Hz | Muscle tone |
| ECG | ECG | 125 Hz | Heart rate |
| Respiratory | Flow, Effort | 25 Hz | Breathing |

## Sleep Stage Encoding

Both Python and MATLAB use consistent sleep stage encoding:

- **0 = Wake** (red)
- **1 = N1** (orange) - Light sleep
- **2 = N2** (green) - Light sleep
- **3 = N3** (blue) - Deep sleep
- **4 = REM** (purple) - Rapid Eye Movement

## Example Workflow

### Python

```python
from src.visualization import plot_sample_epoch, plot_hypnogram

# Load and visualize a recording
edf_file = 'data/training/subject01_night1.edf'
xml_file = 'data/training/subject01_night1.xml'

# 1. Plot hypnogram to see overall sleep pattern
plot_hypnogram(xml_file)

# 2. Identify interesting epochs (e.g., REM vs N3)
# 3. Plot signals from those epochs
plot_sample_epoch(edf_file, epoch_idx=150)  # Example: REM epoch
plot_sample_epoch(edf_file, epoch_idx=300)  # Example: N3 epoch
```

### MATLAB

```matlab
% Load and visualize a recording
edf_file = 'data/training/subject01_night1.edf';
xml_file = 'data/training/subject01_night1.xml';

% 1. Plot hypnogram to see overall sleep pattern
plot_hypnogram(xml_file);

% 2. Identify interesting epochs
% 3. Plot signals from those epochs
plot_sample_epoch(edf_file, 150);  % REM epoch
plot_sample_epoch(edf_file, 300);  % N3 epoch
```

## Tips for Signal Analysis

1. **Compare sleep stages**: Plot epochs from different sleep stages to see characteristic patterns:
   - **Wake**: High-frequency EEG, frequent eye movements, high EMG
   - **N1**: Theta waves (4-8 Hz), slow eye movements
   - **N2**: Sleep spindles (12-14 Hz) and K-complexes
   - **N3**: Delta waves (0.5-4 Hz), low frequency high amplitude
   - **REM**: Mixed frequency EEG, rapid eye movements, low EMG

2. **Artifacts**: Look for artifacts in the signals:
   - Eye blinks in EEG (especially frontal channels)
   - Movement artifacts in EMG
   - Electrode disconnections (flat lines)

3. **Signal quality**: Check sampling frequencies match expected values from the study

4. **Hypnogram patterns**: Normal sleep shows cyclic patterns (~90 min cycles) with deeper sleep early and more REM later

## Requirements

### Python
- `pyedflib`: For reading EDF files
- `matplotlib`: For plotting
- `numpy`: For numerical operations

Install: `pip install pyedflib matplotlib numpy`

### MATLAB
- Biosig toolbox or EEGLAB (for EDF reading)
- Built-in XML parser (included with MATLAB)

## Troubleshooting

**Python: "Error reading EDF file"**
- Check file path is correct
- Ensure `pyedflib` is installed: `pip install pyedflib`
- Verify EDF file is not corrupted (check file size > 1KB)

**MATLAB: "edfread function not found"**
- Install Biosig toolbox: http://biosig.sourceforge.net/
- Or use EEGLAB's `pop_biosig()` function
- Or implement custom EDF reader

**XML parsing errors**
- Ensure XML file matches Compumedics format
- Check XML is well-formed (no corrupted tags)
- Verify sleep stage annotations exist in the file

## Integration with Pipeline

These visualization functions are standalone and can be used independently of the main training pipeline. They are useful for:

1. **Data exploration**: Understanding signal characteristics before preprocessing
2. **Debugging**: Checking if preprocessing is working correctly
3. **Feature validation**: Seeing why certain features may be effective
4. **Results interpretation**: Understanding classification errors by looking at signals
5. **Report generation**: Creating figures for documentation

You can call these functions from the main pipeline or use them interactively in Jupyter notebooks (Python) or MATLAB command window.
