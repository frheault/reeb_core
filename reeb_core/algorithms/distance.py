import networkx as nx
import numpy as np
from copy import deepcopy
from functools import wraps
from scipy.spatial import KDTree as SPKDTree


def compute_node_features(G):
    degree_centrality = nx.degree_centrality(G)

    try:
        closeness_centrality = nx.closeness_centrality(G)
    except Exception:
        closeness_centrality = {node: 0.0 for node in G.nodes}

    try:
        betweenness_centrality = nx.betweenness_centrality(G)
    except Exception:
        betweenness_centrality = {node: 0.0 for node in G.nodes}

    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector_centrality = {node: 0.0 for node in G.nodes}

    clustering_coefficient = nx.clustering(G)

    features = {}
    for node in G.nodes:
        features[node] = [
            degree_centrality.get(node, 0.0),
            closeness_centrality.get(node, 0.0),
            betweenness_centrality.get(node, 0.0),
            eigenvector_centrality.get(node, 0.0),
            clustering_coefficient.get(node, 0.0)
        ]
    return features


def annotate_graph(graph, node_positions):
    for e in graph.edges:
        n1, n2 = e
        pos1, pos2 = np.array(node_positions[n1]), np.array(node_positions[n2])
        graph.nodes[n1]["position"] = pos1
        graph.nodes[n2]["position"] = pos2
        diff = pos1 - pos2
        graph.edges[e]["distance"] = np.sqrt(
            diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])


def merge_equivalent(graph, node_annotations):
    equivalences = dict()
    for pos, node in node_annotations.items():
        pos_tuple = tuple(pos)
        if pos_tuple not in equivalences:
            equivalences[pos_tuple] = []
        equivalences[pos_tuple].append(node)

    for eq_group in equivalences.values():
        if len(eq_group) == 1:
            continue
        head, tail = eq_group[0], eq_group[1:]
        for n in tail:
            if n in graph.nodes:
                nx.contracted_nodes(graph, head, n, copy=False)


def annotate_merge(fn):
    @wraps(fn)
    def wrapped(gcmp, gcmp_pos, gref, gref_pos, *args, **kwargs):
        merge_equivalent(gcmp, gcmp_pos)
        annotate_graph(gcmp, gcmp_pos)

        merge_equivalent(gref, gref_pos)
        annotate_graph(gref, gref_pos)

        return fn(gcmp, gref, *args, **kwargs)
    return wrapped


def max_cost_score(node_score, edge_weight_score, edge_dist_score, gcmp, gref, eps, alpha, delta):
    sub_rad = 2 * eps
    max_node_ins_cost = sub_rad * len(gref.nodes)
    max_node_del_cost = sub_rad * len(gcmp.nodes)

    max_edge_ins_cost = sum(attrs.get("weight", 0.0)
                            for (_, _, attrs) in gref.edges(data=True))
    max_edge_del_cost = sum(attrs.get("weight", 0.0)
                            for (_, _, attrs) in gcmp.edges(data=True))

    max_score = max_node_ins_cost + max_node_del_cost + \
        max_edge_ins_cost + max_edge_del_cost
    if max_score == 0:
        return 0.0

    given_score = node_score
    if edge_dist_score is not None:
        given_score += edge_dist_score
    if edge_weight_score is not None:
        given_score += edge_weight_score

    return given_score / max_score


def _find_closest(n, gref_tree, gref_positions, gref_ids, gref_attrs_sorted,
                  counterpart_nodes, valid_cand, dist_fn_sq, gref):
    """Find the closest valid gref node to the given gcmp node using KDTree.

    Falls back to linear scan if all KDTree candidates are already matched.
    """
    default = (None, {"position": np.array([np.inf, np.inf, np.inf])})

    if gref_tree is None:
        return default

    pos = n[1]["position"]
    # Query enough neighbors to guarantee finding one not in counterpart_nodes
    k = min(len(counterpart_nodes) + 1, len(gref_ids))
    dists, indices = gref_tree.query(pos, k=k)

    # When k=1, scipy returns scalars; normalize to arrays
    if k == 1:
        dists = [dists]
        indices = [indices]

    for idx in indices:
        nid = gref_ids[idx]
        if nid not in counterpart_nodes:
            return (nid, gref.nodes[nid])

    # All k candidates taken — fall back to full linear scan
    return min(filter(valid_cand, gref_attrs_sorted),
               key=lambda m: dist_fn_sq(m, n),
               default=default)


def distance(gcmp, gref, eps, alpha, delta, sub_rad=None, scoring_func=None, transform=False, ins_cost=None, mode="topology"):
    """
    Intakes two graphs, gcmp and gref, and computes topological or Siminet distances between them.

    Parameters
    ----------
    gcmp, gref : networkx.Graph
        Comparison and reference graphs.
    eps, alpha, delta : float
        Reeb parameters.
    sub_rad : float, optional
        Radius of substitution (defaults to 2 * eps).
    scoring_func : callable, optional
        Custom scoring function normalization.
    transform : bool, optional
        If True, transforms and returns a modified copy of gcmp.
    ins_cost : float, optional
        Node insertion cost.
    mode : str
        either "topology" (default, includes centrality features) or "siminet" (includes edge weights/lengths).
    """
    if len(gref.nodes) == 0:
        if scoring_func is None:
            return (0.0, 0.0, 0.0) if mode == "siminet" else (0.0, 0.0)
        return 0.0

    if scoring_func is None:
        if mode == "siminet":
            def scoring_func(n, ew, ed, gc, gr, eps, alpha,
                             delta): return (n, ew, ed)
        else:
            def scoring_func(n, ew, ed, gc, gr, eps,
                             alpha, delta): return (n, ed)

    if sub_rad is None:
        sub_rad = 2 * eps

    assert eps < sub_rad

    copy = deepcopy(gcmp) if transform else None

    def dist_fn_sq(p, q):
        diff = p[1]["position"] - q[1]["position"]
        return diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]

    equivalency_mapping = dict()
    counterpart_nodes = set()

    node_score = 0
    if ins_cost is None:
        ins_cost = sub_rad

    freq_table = {'equivalency': 0, 'substitution': 0, 'deletion': 0}

    def valid_cand(nde): return nde[0] not in counterpart_nodes

    gcmp_attrs_sorted = sorted(gcmp.nodes(data=True), key=lambda n: n[0])
    gref_attrs_sorted = sorted(gref.nodes(data=True), key=lambda n: n[0])

    # Build KDTree from gref positions for fast nearest-neighbor queries
    gref_positions = np.array([attrs["position"]
                              for _, attrs in gref_attrs_sorted])
    gref_ids = [nid for nid, _ in gref_attrs_sorted]
    gref_tree = SPKDTree(gref_positions) if len(gref_positions) > 0 else None

    if mode == "topology":
        feature_ref = compute_node_features(gref)
        feature_cmp = compute_node_features(gcmp)
        network_score = 0

        for n in gcmp_attrs_sorted:
            closest = _find_closest(n, gref_tree, gref_positions, gref_ids, gref_attrs_sorted,
                                    counterpart_nodes, valid_cand, dist_fn_sq, gref)
            if closest[0] is not None:
                diff = n[1]["position"] - closest[1]["position"]
                d = np.sqrt(diff[0]*diff[0] + diff[1]
                            * diff[1] + diff[2]*diff[2])
            else:
                d = np.inf

            if closest[0] is not None:
                d_nm = np.linalg.norm(
                    np.array(feature_cmp[n[0]]) - np.array(feature_ref[closest[0]]))
            else:
                d_nm = 0.0

            if d <= eps:
                freq_table['equivalency'] += 1
                equivalency_mapping[closest[0]] = n[0]
                counterpart_nodes.add(closest[0])
                if transform:
                    copy.nodes[n[0]]["position"] = closest[1]["position"]
            elif d <= sub_rad:
                freq_table['substitution'] += 1
                counterpart_nodes.add(closest[0])
                node_score += d
                network_score += d_nm
                if transform:
                    copy.nodes[n[0]]["position"] = closest[1]["position"]
            else:
                freq_table['deletion'] += 1
                node_score += ins_cost
                d_mean = np.mean(np.array(feature_cmp[n[0]]))
                network_score += d_mean
                if transform:
                    copy.remove_node(n[0])

        not_found = gref.nodes - counterpart_nodes
        freq_table['insertion'] = len(not_found)
        node_score += ins_cost * len(not_found)
        for nf in not_found:
            network_score += np.mean(np.array(feature_ref[nf]))

        if transform:
            for n in not_found:
                copy.add_node(n, **gref.nodes[n])

        edge_weight_score = None
        edge_dist_score = network_score
        node_score = node_score / len(gref.nodes) + network_score

    else:  # mode == "siminet"
        for n in gcmp_attrs_sorted:
            closest = _find_closest(n, gref_tree, gref_positions, gref_ids, gref_attrs_sorted,
                                    counterpart_nodes, valid_cand, dist_fn_sq, gref)
            if closest[0] is not None:
                diff = n[1]["position"] - closest[1]["position"]
                d = np.sqrt(diff[0]*diff[0] + diff[1]
                            * diff[1] + diff[2]*diff[2])
            else:
                d = np.inf

            if d <= eps:
                freq_table['equivalency'] += 1
                equivalency_mapping[closest[0]] = n[0]
                counterpart_nodes.add(closest[0])
                if transform:
                    copy.nodes[n[0]]["position"] = closest[1]["position"]
            elif d <= sub_rad:
                freq_table['substitution'] += 1
                counterpart_nodes.add(closest[0])
                node_score += d
                if transform:
                    copy.nodes[n[0]]["position"] = closest[1]["position"]
            else:
                freq_table['deletion'] += 1
                node_score += ins_cost
                if transform:
                    copy.remove_node(n[0])

        not_found = gref.nodes - counterpart_nodes
        freq_table['insertion'] = len(not_found)
        node_score += ins_cost * len(not_found)

        if transform:
            for n in not_found:
                copy.add_node(n, **gref.nodes[n])

        edge_weight_score = 0
        edge_dist_score = 0
        counterpart_edges = set()

        for e in gref.edges(data=True):
            n1, n2, ref_data = e
            wt = 0
            edist = 0
            add_edge = False

            if n1 in equivalency_mapping and n2 in equivalency_mapping:
                pce = (equivalency_mapping[n1], equivalency_mapping[n2])
                if pce in gcmp.edges:
                    wt = gcmp.edges[pce]["weight"]
                    edist = gcmp.edges[pce]["distance"]
                    counterpart_edges.add(pce)
                else:
                    add_edge = True
            else:
                add_edge = True

            raw_wt_diff = abs(ref_data["weight"] - wt)
            raw_dist_diff = abs(ref_data["distance"] - edist)

            edge_weight_score += raw_wt_diff if raw_wt_diff > (
                delta / len(gref.edges)) else 0
            edge_dist_score += raw_dist_diff if raw_dist_diff > alpha else 0

            if transform and add_edge:
                copy.add_edge(n1, n2, **ref_data)

        lone_edges = gcmp.edges - counterpart_edges
        weight_deletion_score = sum(
            gcmp.edges[e]["weight"] for e in lone_edges)
        dist_deletion_score = sum(
            gcmp.edges[e]["distance"] for e in lone_edges)

        edge_weight_score += weight_deletion_score
        edge_dist_score += dist_deletion_score

    final_score = scoring_func(
        node_score, edge_weight_score, edge_dist_score, gcmp, gref, eps, alpha, delta)
    if transform:
        return final_score, copy
    return final_score
