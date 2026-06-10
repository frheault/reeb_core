"""
Unit tests for event detection logic in events.py.

Tests verify that find_connect_disconnect_events correctly identifies connect
and disconnect events between pairs of synthetic streamlines, and that
alpha filtering removes spatially close event pairs.
"""
import pytest
import numpy as np
from reeb_core.algorithms.events import Event, find_connect_disconnect_events, check_epsilon_distance


class TestCheckEpsilonDistance:
    """Tests for the point-wise epsilon distance check."""

    def test_points_within_distance(self):
        """Points closer than sqrt(eps_sq) should return True."""
        assert check_epsilon_distance(np.array([0, 0, 0], dtype=np.float64), np.array([
                                      0.5, 0, 0], dtype=np.float64), 1.0) is True

    def test_points_outside_distance(self):
        """Points farther than sqrt(eps_sq) should return False."""
        assert check_epsilon_distance(np.array([0, 0, 0], dtype=np.float64), np.array([
                                      2, 0, 0], dtype=np.float64), 1.0) is False

    def test_exact_boundary_inclusive(self):
        """Points at exactly eps_sq distance should return True (<= comparison)."""
        # dist_sq = 1.0, eps_sq = 1.0 → exactly on boundary
        assert check_epsilon_distance(np.array([0, 0, 0], dtype=np.float64), np.array([
                                      1, 0, 0], dtype=np.float64), 1.0) is True

    def test_3d_diagonal_distance(self):
        """Distance check works correctly across all three axes."""
        # dist = sqrt(0.3^2 + 0.4^2 + 0^2) = 0.5, dist_sq = 0.25
        assert check_epsilon_distance(np.array([0, 0, 0], dtype=np.float64), np.array([
                                      0.3, 0.4, 0], dtype=np.float64), 0.26) is True
        assert check_epsilon_distance(np.array([0, 0, 0], dtype=np.float64), np.array([
                                      0.3, 0.4, 0], dtype=np.float64), 0.24) is False

    def test_identical_points(self):
        """Identical points have distance 0, always within eps."""
        assert check_epsilon_distance(np.array([5.0, 3.0, 1.0], dtype=np.float64), np.array(
            [5.0, 3.0, 1.0], dtype=np.float64), 0.0) is True

    def test_works_with_numpy_arrays(self):
        """Function works with numpy arrays, not just lists."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.5, 0.0, 0.0])
        assert check_epsilon_distance(p1, p2, 1.0) == True


class TestFindConnectDisconnectEvents:
    """Tests for the main event detection function."""

    def test_parallel_lines_fully_within_eps(self):
        """Two parallel lines always within eps → connect at start, no disconnect."""
        n = 10
        t1 = [[i, 0, 0] for i in range(n)]
        t2 = [[i, 0.3, 0] for i in range(n)]
        eps = 1.0
        alpha = 50.0  # large alpha to prevent any filtering

        events_t1, events_t2 = find_connect_disconnect_events(
            0, 1, t1, t2, eps, alpha)

        # Should have a connect event at the very start
        assert 0 in events_t1, "Expected connect event at index 0"
        connect_events = [e for e in events_t1[0] if e.event == "connect"]
        assert len(connect_events) == 1
        assert connect_events[0].trajectory == 1

        # No disconnect since connection spans the entire streamline
        all_disconnects = []
        for events in events_t1.values():
            all_disconnects.extend(
                e for e in events if e.event == "disconnect")
        assert len(all_disconnects) == 0, \
            "Parallel lines fully within eps should produce no disconnect events"

    def test_no_interaction(self):
        """Two lines that never come within eps produce no events."""
        n = 10
        t1 = [[i, 0, 0] for i in range(n)]
        t2 = [[i, 10, 0] for i in range(n)]
        eps = 1.0
        alpha = 2.0

        events_t1, events_t2 = find_connect_disconnect_events(
            0, 1, t1, t2, eps, alpha)
        assert len(events_t1) == 0
        assert len(events_t2) == 0

    def test_converge_then_diverge(self):
        """Lines that converge then diverge produce connect and disconnect events."""
        n = 20
        t1 = [[i, 0, 0] for i in range(n)]
        # Far away for i<5 and i>=15, close for 5<=i<15
        t2 = [[i, (5.0 if i < 5 or i >= 15 else 0.3), 0] for i in range(n)]
        eps = 1.0
        alpha = 2.0

        events_t1, events_t2 = find_connect_disconnect_events(
            0, 1, t1, t2, eps, alpha)

        # Collect all events from events_t1
        event_types_t1 = []
        for key in sorted(events_t1.keys()):
            for e in events_t1[key]:
                event_types_t1.append((key, e.event))

        connect_keys = [k for k, ev in event_types_t1 if ev == "connect"]
        disconnect_keys = [k for k, ev in event_types_t1 if ev == "disconnect"]

        assert len(connect_keys) >= 1, "Expected at least one connect event"
        assert len(disconnect_keys) >= 1, "Expected at least one disconnect event"
        assert min(connect_keys) < min(disconnect_keys), \
            "Connect should occur at an earlier index than disconnect"

    def test_symmetric_trajectory_references(self):
        """Both trajectory dicts reference each other's IDs correctly."""
        n = 20
        t1 = [[i, 0, 0] for i in range(n)]
        t2 = [[i, (5.0 if i < 5 or i >= 15 else 0.3), 0] for i in range(n)]
        eps = 1.0
        alpha = 2.0

        events_t1, events_t2 = find_connect_disconnect_events(
            0, 1, t1, t2, eps, alpha)

        # Both should have events
        assert len(events_t1) > 0
        assert len(events_t2) > 0

        # t1's events should reference trajectory 1
        for events in events_t1.values():
            for e in events:
                if e.event in ("connect", "disconnect"):
                    assert e.trajectory == 1, \
                        f"t1 event should reference trajectory 1, got {e.trajectory}"

        # t2's events should reference trajectory 0
        for events in events_t2.values():
            for e in events:
                if e.event in ("connect", "disconnect"):
                    assert e.trajectory == 0, \
                        f"t2 event should reference trajectory 0, got {e.trajectory}"

    def test_alpha_filtering_removes_close_events(self):
        """Events on t1 closer than alpha apart are removed by alpha filtering.

        When streamlines only interact for 2 consecutive points (indices 8-9),
        the connect event (index 8) and disconnect event (index 10) on t1 are
        spatially only 2 units apart. With alpha=3.0, this falls below the
        threshold, so alpha filtering removes both events from events_t1.
        """
        n = 20
        t1 = [[i, 0, 0] for i in range(n)]
        # Only indices 8 and 9 are within eps
        t2 = [[i, (0.3 if 8 <= i <= 9 else 5.0), 0] for i in range(n)]
        eps = 1.0
        alpha = 3.0  # t1[8]→t1[10] distance = 2.0 < alpha = 3.0

        events_t1, events_t2 = find_connect_disconnect_events(
            0, 1, t1, t2, eps, alpha)

        # t1 event positions (index 8 and 10) are 2.0 apart < alpha=3.0
        # Alpha filtering removes both events from events_t1
        assert len(events_t1) == 0, \
            "Alpha filtering should remove events when consecutive positions are closer than alpha"

    def test_event_object_attributes(self):
        """Event objects store the correct attributes."""
        e = Event("connect", trajectory=5, t=10)
        assert e.event == "connect"
        assert e.trajectory == 5
        assert e.t == 10

        e2 = Event("appear")
        assert e2.event == "appear"
        assert e2.trajectory is None
        assert e2.t is None

    def test_single_point_streamlines(self):
        """Streamlines of length 1 that are within eps produce a connect event."""
        t1 = [[0, 0, 0]]
        t2 = [[0, 0.5, 0]]
        eps = 1.0
        alpha = 50.0

        events_t1, events_t2 = find_connect_disconnect_events(
            0, 1, t1, t2, eps, alpha)

        # Should have a connect event
        assert 0 in events_t1
        connect_events = [e for e in events_t1[0] if e.event == "connect"]
        assert len(connect_events) == 1
