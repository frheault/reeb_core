import pytest


def test_imports():
    try:
        import reeb_core
        from reeb_core import Event, find_connect_disconnect_events, construct_robust_reeb, distance, max_cost_score
        from reeb_core.utils import load_streamlines
        from reeb_core.cli import visualizer

        print("All imports succeeded!")
    except ImportError as e:
        pytest.fail(f"Failed to import reeb_core package: {e}")
