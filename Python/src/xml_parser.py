"""
XML Parser for Sleep Stage Annotations

This module provides functions to parse Compumedics XML annotation files
containing sleep stage labels.

Format: http://www.compumedics.com.au/profilingxml/ProfilingViewer_ProfilingDatabase.xsd
"""

import xml.etree.ElementTree as ET
import numpy as np


def parse_xml_annotations(xml_file_path):
    """
    Parse XML annotation file to extract sleep stage labels.

    The XML file contains ScoredEvent elements with sleep stage classifications
    according to the SDO (Sleep Domain Ontology) standard.

    Args:
        xml_file_path (str): Path to XML annotation file.

    Returns:
        dict: Dictionary containing:
            - 'events': List of all events (dict with 'concept', 'start', 'duration')
            - 'stages': List of sleep stage events only
            - 'epoch_length': Epoch duration in seconds (typically 30)

    Raises:
        FileNotFoundError: If XML file doesn't exist.
        ET.ParseError: If XML file is malformed.
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
    except FileNotFoundError:
        raise FileNotFoundError(f"XML file not found: {xml_file_path}")
    except ET.ParseError as e:
        raise ET.ParseError(f"Failed to parse XML file {xml_file_path}: {e}")

    # Extract epoch length (default 30 seconds)
    epoch_length = 30
    epoch_elements = root.findall('.//EpochLength')
    if epoch_elements:
        epoch_length = float(epoch_elements[0].text)

    # Parse all scored events
    events = []
    stages = []

    # Sleep stage concept mappings (SDO ontology)
    stage_map = {
        'SDO:NonRapidEyeMovementSleep-N1': 1,  # N1
        'SDO:NonRapidEyeMovementSleep-N2': 2,  # N2
        'SDO:NonRapidEyeMovementSleep-N3': 3,  # N3
        'SDO:NonRapidEyeMovementSleep-N4': 3,  # N4 (rare, usually mapped to N3)
        'SDO:RapidEyeMovementSleep': 4,        # REM
        'SDO:WakeState': 0                      # Wake
    }

    scored_events = root.findall('.//ScoredEvent')

    for event in scored_events:
        # Extract event concept (sleep stage or other event type)
        concept_elem = event.find('EventConcept')
        if concept_elem is None:
            continue

        concept = concept_elem.text.strip()

        # Extract start time and duration
        start_elem = event.find('Start')
        duration_elem = event.find('Duration')

        if start_elem is None or duration_elem is None:
            continue

        start_time = float(start_elem.text)
        duration = float(duration_elem.text)

        # Store all events
        event_dict = {
            'concept': concept,
            'start': start_time,
            'duration': duration
        }

        # Extract additional fields if present
        if event.find('Desaturation') is not None:
            event_dict['desaturation'] = float(event.find('Desaturation').text)
        if event.find('SpO2Nadir') is not None:
            event_dict['spo2_nadir'] = float(event.find('SpO2Nadir').text)
        if event.find('Text') is not None:
            event_dict['text'] = event.find('Text').text

        events.append(event_dict)

        # If this is a sleep stage event, add to stages list
        if concept in stage_map:
            stage_event = {
                'stage': stage_map[concept],
                'start': start_time,
                'duration': duration
            }
            stages.append(stage_event)

    return {
        'events': events,
        'stages': stages,
        'epoch_length': epoch_length
    }


def create_epoch_labels(stages, total_duration, epoch_length=30):
    """
    Convert variable-duration stage events to fixed-length epoch labels.

    Args:
        stages (list): List of stage events from parse_xml_annotations()
        total_duration (float): Total recording duration in seconds
        epoch_length (float): Epoch duration in seconds (default 30)

    Returns:
        np.ndarray: Array of shape (n_epochs,) with integer labels 0-4

    Example:
        >>> stages = [
        ...     {'stage': 0, 'start': 0, 'duration': 60},    # Wake for 60s
        ...     {'stage': 2, 'start': 60, 'duration': 120}   # N2 for 120s
        ... ]
        >>> labels = create_epoch_labels(stages, total_duration=180, epoch_length=30)
        >>> print(labels)  # [0, 0, 2, 2, 2, 2]
    """
    n_epochs = int(np.ceil(total_duration / epoch_length))
    labels = np.zeros(n_epochs, dtype=int)

    for stage_event in stages:
        stage = stage_event['stage']
        start = stage_event['start']
        duration = stage_event['duration']

        # Calculate which epochs this stage covers
        start_epoch = int(start / epoch_length)
        end_time = start + duration
        end_epoch = int(np.ceil(end_time / epoch_length))

        # Assign stage label to all epochs in this range
        labels[start_epoch:end_epoch] = stage

    return labels


def validate_annotations(xml_file_path, edf_duration):
    """
    Validate that XML annotations match the EDF recording duration.

    Args:
        xml_file_path (str): Path to XML annotation file
        edf_duration (float): Duration of EDF recording in seconds

    Returns:
        dict: Validation results with keys:
            - 'valid': bool, True if annotations cover the entire recording
            - 'annotation_duration': float, total duration of annotations
            - 'coverage': float, percentage of recording covered (0-100)
            - 'gaps': list of time gaps in annotations
            - 'overlaps': list of overlapping annotations

    Example:
        >>> results = validate_annotations('R1.xml', 32400)  # 9 hours
        >>> print(f"Coverage: {results['coverage']:.1f}%")
    """
    parsed = parse_xml_annotations(xml_file_path)
    stages = parsed['stages']

    if not stages:
        return {
            'valid': False,
            'annotation_duration': 0,
            'coverage': 0,
            'gaps': [],
            'overlaps': [],
            'message': 'No sleep stage annotations found'
        }

    # Sort stages by start time
    stages = sorted(stages, key=lambda x: x['start'])

    # Calculate total annotated duration
    last_event = stages[-1]
    annotation_duration = last_event['start'] + last_event['duration']

    # Check for gaps
    gaps = []
    for i in range(len(stages) - 1):
        current_end = stages[i]['start'] + stages[i]['duration']
        next_start = stages[i + 1]['start']

        if next_start > current_end:
            gaps.append({
                'start': current_end,
                'end': next_start,
                'duration': next_start - current_end
            })

    # Check for overlaps
    overlaps = []
    for i in range(len(stages) - 1):
        current_end = stages[i]['start'] + stages[i]['duration']
        next_start = stages[i + 1]['start']

        if next_start < current_end:
            overlaps.append({
                'event1_end': current_end,
                'event2_start': next_start,
                'overlap_duration': current_end - next_start
            })

    # Calculate coverage
    coverage = (annotation_duration / edf_duration) * 100 if edf_duration > 0 else 0

    return {
        'valid': len(gaps) == 0 and len(overlaps) == 0 and coverage >= 99,
        'annotation_duration': annotation_duration,
        'edf_duration': edf_duration,
        'coverage': coverage,
        'gaps': gaps,
        'overlaps': overlaps,
        'n_stages': len(stages)
    }


if __name__ == '__main__':
    # Example usage
    import sys

    if len(sys.argv) > 1:
        xml_file = sys.argv[1]

        print(f"Parsing XML file: {xml_file}")
        parsed = parse_xml_annotations(xml_file)

        print(f"\nEpoch length: {parsed['epoch_length']} seconds")
        print(f"Total stage events: {len(parsed['stages'])}")
        print(f"Total events (all types): {len(parsed['events'])}")

        # Show stage distribution
        if parsed['stages']:
            stages_array = np.array([s['stage'] for s in parsed['stages']])
            stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']

            print("\nStage distribution:")
            for stage_num in range(5):
                count = np.sum(stages_array == stage_num)
                if count > 0:
                    pct = (count / len(stages_array)) * 100
                    print(f"  {stage_names[stage_num]}: {count} events ({pct:.1f}%)")

            # Calculate total duration
            total_duration = sum(s['duration'] for s in parsed['stages'])
            print(f"\nTotal annotated duration: {total_duration:.0f} seconds ({total_duration/3600:.2f} hours)")
    else:
        print("Usage: python xml_parser.py <xml_file_path>")
