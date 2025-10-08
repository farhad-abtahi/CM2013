"""
Basic tests to validate configuration and setup.
Students can use these as examples for more comprehensive testing.
"""
import pytest
import os
import sys

# Add the parent directory to the path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_data_directories_exist():
    """Test that essential data directories exist."""
    import config

    # Check that data directory exists (relative to Python/ directory)
    assert os.path.exists(config.DATA_DIR), f"Data directory not found: {config.DATA_DIR}"
    assert os.path.exists(config.SAMPLE_DIR), f"Sample directory not found: {config.SAMPLE_DIR}"

def test_sample_files_exist():
    """Test that sample files exist for jumpstart."""
    import config

    dummy_edf = os.path.join(config.SAMPLE_DIR, "dummy.edf")
    dummy_xml = os.path.join(config.SAMPLE_DIR, "dummy.xml")

    assert os.path.exists(dummy_edf), f"Sample EDF file not found: {dummy_edf}"
    assert os.path.exists(dummy_xml), f"Sample XML file not found: {dummy_xml}"

def test_iteration_config_validity():
    """Test that CURRENT_ITERATION produces valid configuration."""
    import config

    # Test that iteration is in valid range
    assert 1 <= config.CURRENT_ITERATION <= 4, f"Invalid iteration: {config.CURRENT_ITERATION}"

    # Test that classifier type is set
    assert hasattr(config, 'CLASSIFIER_TYPE'), "CLASSIFIER_TYPE not set in config"
    assert config.CLASSIFIER_TYPE in ['knn', 'svm', 'random_forest'], f"Invalid classifier type: {config.CLASSIFIER_TYPE}"

def test_cache_directory_creation():
    """Test that cache directory is created properly."""
    import config

    # Cache directory should be created by config.py
    assert os.path.exists(config.CACHE_DIR), f"Cache directory not created: {config.CACHE_DIR}"
    assert os.path.isdir(config.CACHE_DIR), f"Cache path is not a directory: {config.CACHE_DIR}"

if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])