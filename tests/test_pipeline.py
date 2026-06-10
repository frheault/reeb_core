import pytest
import numpy as np
import networkx as nx
from reeb_core import construct_robust_reeb, distance, max_cost_score
from reeb_core.algorithms.distance import annotate_graph


def test_reeb_construction_and_distance():
    # 1. Create mock streamlines
    # Streamline 1: straight line along x-axis
    s1 = np.zeros((40, 3))
    s1[:, 0] = np.linspace(0.0, 39.0, 40)

    # Streamline 2: slightly shifted straight line
    s2 = np.zeros((40, 3))
    s2[:, 0] = np.linspace(0.0, 39.0, 40)
    s2[:, 1] = 0.5

    # Streamline 3: branching streamline (starts with others, splits away)
    s3 = np.zeros((40, 3))
    s3[:, 0] = np.linspace(0.0, 39.0, 40)
    s3[:20, 1] = 0.2
    s3[20:, 1] = np.linspace(0.2, 5.0, 20)

    streamlines = [s1, s2, s3]

    # 2. Run Reeb graph construction (with Tractosearch clustering)
    eps = 2.0
    alpha = 3.0
    delta = 1

    # Run construction with a low clustering threshold so they cluster nicely
    R, node_loc = construct_robust_reeb(
        streamlines,
        eps=eps,
        alpha=alpha,
        delta=delta,
        clustering_threshold=4.0,
        resample_nb=40
    )

    # Verify return types
    assert isinstance(R, nx.Graph)
    assert isinstance(node_loc, dict)

    if len(R.nodes) > 0:
        # Check that coordinates are returned for every node
        for node in R.nodes:
            assert node in node_loc
            assert len(node_loc[node]) == 3

        # 3. Compute distance between graph and copy
        # Annotate graph positions
        annotate_graph(R, node_loc)

        # Test topology mode (ReTrace)
        score_top = distance(
            R, R,
            eps=eps, alpha=alpha, delta=delta,
            scoring_func=max_cost_score,
            mode="topology"
        )
        # Self distance should be 0.0 ideally
        assert score_top == pytest.approx(0.0, abs=1e-5)

        # Test siminet mode (ReeBundle)
        score_sim = distance(
            R, R,
            eps=eps, alpha=alpha, delta=delta,
            scoring_func=max_cost_score,
            mode="siminet"
        )
        assert score_sim == pytest.approx(0.0, abs=1e-5)
