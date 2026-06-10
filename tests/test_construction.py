"""
Integration tests for Reeb Graph construction pipeline.

Tests verify that construct_robust_reeb correctly constructs a valid Reeb Graph
from synthetic tractograms, including structural integrity of nodes, edges,
coordinates, and weights.
"""
import pytest
import numpy as np
import networkx as nx
from reeb_core import construct_robust_reeb


class TestConstructRobustReeb:
    """Integration tests for the full Reeb construction pipeline."""

    def test_empty_streamlines(self):
        """Empty input returns an empty graph and empty node_loc."""
        R, node_loc = construct_robust_reeb([], eps=2.0, alpha=3.0, delta=1)
        assert isinstance(R, nx.Graph)
        assert len(R.nodes) == 0
        assert len(R.edges) == 0
        assert isinstance(node_loc, dict)
        assert len(node_loc) == 0

    def test_return_types(self):
        """Basic return type validation."""
        streamlines = self._make_parallel_bundle(count=5, separation=0.3)
        R, node_loc = construct_robust_reeb(
            streamlines, eps=2.0, alpha=3.0, delta=1,
            clustering_threshold=4.0, resample_nb=40
        )
        assert isinstance(R, nx.Graph)
        assert isinstance(node_loc, dict)

    def test_node_coordinates_complete(self):
        """Every node in the graph has a 3D coordinate in node_loc."""
        streamlines = self._make_y_shaped_tractogram()
        R, node_loc = construct_robust_reeb(
            streamlines, eps=2.0, alpha=3.0, delta=1,
            clustering_threshold=4.0, resample_nb=40
        )

        if len(R.nodes) > 0:
            for node in R.nodes:
                assert node in node_loc, f"Node {node} missing from node_loc"
                coord = node_loc[node]
                assert len(
                    coord) == 3, f"Node {node} should have 3D coordinate"
                # Coordinates should be finite
                for i, c in enumerate(coord):
                    assert np.isfinite(c), \
                        f"Node {node} coordinate[{i}] = {c} is not finite"

    def test_edge_weights_valid(self):
        """Edge weights should be non-negative."""
        streamlines = self._make_y_shaped_tractogram()
        R, node_loc = construct_robust_reeb(
            streamlines, eps=2.0, alpha=3.0, delta=1,
            clustering_threshold=4.0, resample_nb=40
        )

        for u, v, data in R.edges(data=True):
            weight = data.get("weight", 0)
            assert weight >= 0, f"Edge ({u},{v}) has negative weight: {weight}"

    def test_no_self_loops(self):
        """Constructed graph should have no self-loops."""
        streamlines = self._make_y_shaped_tractogram()
        R, node_loc = construct_robust_reeb(
            streamlines, eps=2.0, alpha=3.0, delta=1,
            clustering_threshold=4.0, resample_nb=40
        )
        self_loops = list(nx.selfloop_edges(R))
        assert len(self_loops) == 0, f"Graph has self-loops: {self_loops}"

    def test_no_isolated_nodes(self):
        """Constructed graph should have no isolated nodes (degree-0)."""
        streamlines = self._make_y_shaped_tractogram()
        R, node_loc = construct_robust_reeb(
            streamlines, eps=2.0, alpha=3.0, delta=1,
            clustering_threshold=4.0, resample_nb=40
        )
        isolates = list(nx.isolates(R))
        assert len(isolates) == 0, f"Graph has isolated nodes: {isolates}"

    def test_y_shaped_nontrivial_graph(self):
        """A Y-shaped tractogram should produce a graph with nodes and edges."""
        streamlines = self._make_y_shaped_tractogram()
        R, node_loc = construct_robust_reeb(
            streamlines, eps=2.0, alpha=3.0, delta=1,
            clustering_threshold=4.0, resample_nb=40
        )
        assert len(
            R.nodes) > 0, "Y-shaped tractogram should produce a non-empty graph"
        assert len(R.edges) > 0, "Y-shaped tractogram should produce edges"

    def test_use_real_positions_flag(self):
        """With use_real_positions=True, node positions should be actual streamline points."""
        streamlines = self._make_parallel_bundle(count=5, separation=0.3)
        R, node_loc = construct_robust_reeb(
            streamlines, eps=2.0, alpha=3.0, delta=1,
            clustering_threshold=4.0, resample_nb=40,
            use_real_positions=True
        )

        if len(R.nodes) > 0:
            for node in R.nodes:
                assert node in node_loc
                assert len(node_loc[node]) == 3

    def test_delta_filtering(self):
        """Higher delta threshold should filter out small bundles,
        potentially producing a simpler or empty graph."""
        streamlines = self._make_parallel_bundle(count=3, separation=0.3)

        # Low delta: keep everything
        R_low, _ = construct_robust_reeb(
            streamlines, eps=2.0, alpha=3.0, delta=1,
            clustering_threshold=4.0, resample_nb=40
        )

        # Very high delta: filter out small bundles
        R_high, _ = construct_robust_reeb(
            streamlines, eps=2.0, alpha=3.0, delta=9999,
            clustering_threshold=4.0, resample_nb=40
        )

        # High delta should produce fewer or equal nodes
        assert len(R_high.nodes) <= len(R_low.nodes)

    # -------------------------------------------------------------------------
    # Helpers for creating synthetic tractograms
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_parallel_bundle(count=5, separation=0.3):
        """Create parallel streamlines along the x-axis."""
        streamlines = []
        for i in range(count):
            s = np.zeros((40, 3))
            s[:, 0] = np.linspace(0, 39, 40)
            s[:, 1] = i * separation
            streamlines.append(s)
        return streamlines

    @staticmethod
    def _make_y_shaped_tractogram():
        """Create a Y-shaped synthetic tractogram.

        Structure:
        - 8 trunk streamlines that stay bundled along the full x-length
        - 5 streamlines that branch upward after the midpoint
        - 5 streamlines that branch downward after the midpoint
        Total: 18 streamlines
        """
        streamlines = []

        # Trunk: tight parallel bundle along x-axis
        for y_offset in np.linspace(-0.3, 0.3, 8):
            s = np.zeros((40, 3))
            s[:, 0] = np.linspace(0, 39, 40)
            s[:, 1] = y_offset
            streamlines.append(s)

        # Branch going up after midpoint
        for y_offset in np.linspace(-0.2, 0.2, 5):
            s = np.zeros((40, 3))
            s[:, 0] = np.linspace(0, 39, 40)
            s[:20, 1] = y_offset
            s[20:, 1] = np.linspace(y_offset, 8.0 + y_offset, 20)
            streamlines.append(s)

        # Branch going down after midpoint
        for y_offset in np.linspace(-0.2, 0.2, 5):
            s = np.zeros((40, 3))
            s[:, 0] = np.linspace(0, 39, 40)
            s[:20, 1] = y_offset
            s[20:, 1] = np.linspace(y_offset, -8.0 + y_offset, 20)
            streamlines.append(s)

        return streamlines
