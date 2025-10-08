"""
Basic pipeline integration tests.
These test the overall flow and can help students debug issues.
"""
import pytest
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_pipeline_imports():
    """Test that all pipeline modules can be imported (even if they're stubs)."""
    try:
        import config
        from src.data_loader import load_training_data
        from src.preprocessing import preprocess
        from src.feature_extraction import extract_features
        from src.feature_selection import select_features
        from src.classification import train_classifier
        from src.visualization import visualize_results
        from src.report import generate_report
        from src.utils import save_cache, load_cache
    except ImportError as e:
        pytest.fail(f"Failed to import pipeline modules: {e}")

def test_data_loading_interface():
    """Test that data loading function has correct interface."""
    from src.data_loader import load_training_data
    import config

    # Test function exists and is callable
    assert callable(load_training_data), "load_training_data should be callable"

    # Note: We don't actually call the function here since it might not be implemented
    # Students should expand this test once they implement the function

def test_cache_utilities_interface():
    """Test that cache utility functions have correct interface."""
    from src.utils import save_cache, load_cache

    assert callable(save_cache), "save_cache should be callable"
    assert callable(load_cache), "load_cache should be callable"

@pytest.mark.integration
def test_full_pipeline_structure():
    """
    Test that main.py can be imported without errors.
    This doesn't run the pipeline but checks the structure is valid.
    """
    try:
        import main
        assert callable(main.main), "main.main should be callable"
    except ImportError as e:
        pytest.fail(f"Failed to import main module: {e}")

if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])