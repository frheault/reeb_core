"""
Unit tests for the graph distance metric in distance.py.

Tests verify that the distance function correctly computes identity distances,
responds to graph perturbations, and handles edge cases like disconnected
and empty graphs.
"""
import pytest
import numpy as np
import networkx as nx
from copy import deepcopy
from reeb_core import distance, max_cost_score
from reeb_core.algorithms.distance import annotate_graph, compute_node_features


def _make_line_graph():
    """Create a simple path graph: 0 -- 1 -- 2 with positions and weights.

    Positions: 0→(0,0,0), 1→(5,0,0), 2→(10,0,0)
    Edge weights: 0.5 each, distances: 5.0 each.
    """
    G = nx.Graph()
    node_pos = {
        0: [0, 0, 0],
        1: [5, 0, 0],
        2: [10, 0, 0],
    }
    G.add_edge(0, 1, weight=0.5)
    G.add_edge(1, 2, weight=0.5)
    annotate_graph(G, node_pos)
    return G, node_pos


def _make_triangle_graph():
    """Create a triangle graph: 0 -- 1 -- 2 -- 0 with positions and weights."""
    G = nx.Graph()
    node_pos = {
        0: [0, 0, 0],
        1: [5, 0, 0],
        2: [2.5, 4, 0],
    }
    G.add_edge(0, 1, weight=0.4)
    G.add_edge(1, 2, weight=0.3)
    G.add_edge(2, 0, weight=0.3)
    annotate_graph(G, node_pos)
    return G, node_pos


class TestIdentityDistance:
    """Self-distance should be 0 for both topology and siminet modes."""

    def test_identity_topology_mode(self):
        """distance(G, G) == 0.0 in topology mode with max_cost_score."""
        G, _ = _make_line_graph()
        score = distance(
            G, G, eps=2.0, alpha=3.0, delta=1,
            scoring_func=max_cost_score, mode="topology"
        )
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_identity_siminet_mode(self):
        """distance(G, G) == 0.0 in siminet mode with max_cost_score."""
        G, _ = _make_line_graph()
        score = distance(
            G, G, eps=2.0, alpha=3.0, delta=1,
            scoring_func=max_cost_score, mode="siminet"
        )
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_identity_topology_raw_scores(self):
        """Raw topology score tuple should be (0, 0) for self-distance."""
        G, _ = _make_line_graph()
        result = distance(G, G, eps=2.0, alpha=3.0, delta=1, mode="topology")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == pytest.approx(0.0, abs=1e-10)
        assert result[1] == pytest.approx(0.0, abs=1e-10)

    def test_identity_siminet_raw_scores(self):
        """Raw siminet score tuple should be (0, 0, 0) for self-distance."""
        G, _ = _make_line_graph()
        result = distance(G, G, eps=2.0, alpha=3.0, delta=1, mode="siminet")
        assert isinstance(result, tuple)
        assert len(result) == 3
        for s in result:
            assert s == pytest.approx(0.0, abs=1e-10)

    def test_identity_triangle(self):
        """Self-distance is 0 for a triangle graph."""
        G, _ = _make_triangle_graph()
        score = distance(
            G, G, eps=2.0, alpha=3.0, delta=1,
            scoring_func=max_cost_score, mode="topology"
        )
        assert score == pytest.approx(0.0, abs=1e-10)


class TestPerturbedDistance:
    """Perturbations to the graph should increase distance."""

    def test_removed_node_increases_distance(self):
        """Removing a node from gcmp should produce distance > 0."""
        G, node_pos = _make_line_graph()

        # Create gcmp with node 2 removed
        G_modified = nx.Graph()
        G_modified.add_edge(0, 1, weight=0.5)
        annotate_graph(G_modified, {0: [0, 0, 0], 1: [5, 0, 0]})

        score = distance(
            G_modified, G, eps=2.0, alpha=3.0, delta=1,
            scoring_func=max_cost_score, mode="topology"
        )
        assert score > 0.0, "Removing a node should increase distance"

    def test_shifted_node_increases_distance(self):
        """Shifting a node position beyond eps should cause substitution cost."""
        G_ref, _ = _make_line_graph()

        # Create gcmp with node 1 shifted in y by more than eps
        G_cmp = nx.Graph()
        node_pos_cmp = {
            0: [0, 0, 0],
            1: [5, 3, 0],   # shifted by 3 in y (> eps=2)
            2: [10, 0, 0],
        }
        G_cmp.add_edge(0, 1, weight=0.5)
        G_cmp.add_edge(1, 2, weight=0.5)
        annotate_graph(G_cmp, node_pos_cmp)

        score = distance(
            G_cmp, G_ref, eps=2.0, alpha=3.0, delta=1,
            scoring_func=max_cost_score, mode="topology"
        )
        assert score > 0.0, \
            "Shifting a node beyond eps should produce non-zero distance"

    def test_added_edge_increases_siminet_distance(self):
        """Extra edge in gcmp should produce non-zero siminet distance."""
        G_ref, _ = _make_line_graph()

        # gcmp adds an extra edge 0--2
        G_cmp = deepcopy(G_ref)
        G_cmp.add_edge(0, 2, weight=0.3, distance=10.0)

        score = distance(
            G_cmp, G_ref, eps=2.0, alpha=3.0, delta=1,
            scoring_func=max_cost_score, mode="siminet"
        )
        assert score > 0.0, "Extra edge should increase siminet distance"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_reference_graph_topology(self):
        """Distance with empty reference graph should return (0.0, 0.0)."""
        G, _ = _make_line_graph()
        G_empty = nx.Graph()

        result = distance(G, G_empty, eps=2.0, alpha=3.0,
                          delta=1, mode="topology")
        assert result == (0.0, 0.0)

    def test_empty_reference_graph_siminet(self):
        """Distance with empty reference graph should return (0.0, 0.0, 0.0)."""
        G, _ = _make_line_graph()
        G_empty = nx.Graph()

        result = distance(G, G_empty, eps=2.0, alpha=3.0,
                          delta=1, mode="siminet")
        assert result == (0.0, 0.0, 0.0)

    def test_empty_reference_with_scoring_func(self):
        """Distance with empty reference and scoring_func returns 0.0."""
        G, _ = _make_line_graph()
        G_empty = nx.Graph()

        result = distance(
            G, G_empty, eps=2.0, alpha=3.0, delta=1,
            scoring_func=max_cost_score
        )
        assert result == 0.0

    def test_transform_returns_modified_graph(self):
        """With transform=True, distance returns (score, modified_graph)."""
        G, _ = _make_line_graph()
        result = distance(
            G, G, eps=2.0, alpha=3.0, delta=1,
            scoring_func=max_cost_score, mode="topology",
            transform=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        score, modified = result
        assert isinstance(modified, nx.Graph)


class TestComputeNodeFeatures:
    """Tests for the node feature computation helper."""

    def test_basic_features(self):
        """compute_node_features returns 5-element feature vector per node."""
        G, _ = _make_line_graph()
        features = compute_node_features(G)
        assert len(features) == 3  # 3 nodes
        for node in G.nodes:
            assert len(features[node]) == 5
            # All centrality values should be finite
            for val in features[node]:
                assert np.isfinite(val)

    def test_disconnected_graph_does_not_crash(self):
        """compute_node_features handles disconnected graphs without crashing."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        # 0--1 and 2--3 are disconnected components

        features = compute_node_features(G)
        assert len(features) == 4
        for node in G.nodes:
            assert len(features[node]) == 5
            for val in features[node]:
                assert np.isfinite(val)

    def test_single_node_graph(self):
        """compute_node_features works on a graph with one isolated node."""
        G = nx.Graph()
        G.add_node(0)

        features = compute_node_features(G)
        assert 0 in features
        assert len(features[0]) == 5
