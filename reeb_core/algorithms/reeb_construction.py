import numpy as np
import networkx as nx
import time
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
from scipy.spatial import KDTree
from tqdm import tqdm
from reeb_core.algorithms.events import Event, find_connect_disconnect_events
from reeb_core.utils.timing import log_timer

# Tractosearch imports
from tractosearch.resampling import resample_slines_to_array
from tractosearch.search import radius_search
from tractosearch.group import connected_components_indices, connected_components_split, group_to_centroid

logger = logging.getLogger("reeb_core")
# Set to DEBUG for detailed timing logs, change to INFO for less verbosity
logger.setLevel(logging.DEBUG)


def distance(p1, p2):
    """
    Computes Euclidean Distance between two 3D points.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


def squared_distance(p1, p2):
    """
    Computes Squared Euclidean Distance between two 3D points.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return dx * dx + dy * dy + dz * dz


def find_closest_real_point(centroid, candidates):
    """
    Finds the point among candidates that is closest to the centroid.
    """
    min_dist = float('inf')
    closest_point = centroid
    for p in candidates:
        d = squared_distance(centroid, p)
        if d < min_dist:
            min_dist = d
            closest_point = p
    return closest_point


def _downsample_streamlines(streamlines, clustering_threshold, resample_nb):
    """Resample streamlines and cluster them using Tractosearch.

    Performs resampling to a uniform number of points, radius-based
    similarity search, connected-component clustering, and centroid
    computation for each cluster.

    Parameters
    ----------
    streamlines : list of numpy arrays
        Raw input streamlines.
    clustering_threshold : float
        MDF distance threshold for Tractosearch clustering.
    resample_nb : int
        Number of points each streamline is resampled to.

    Returns
    -------
    centroid_trk : list of list
        Centroid streamlines (one per cluster).
    cluster_map : dict
        Mapping from centroid index to the number of streamlines in that cluster.
    resample_nb : int
        The resampling count (passed through unchanged).
    """
    with log_timer("resample_slines_to_array"):
        slines_arr = resample_slines_to_array(
            streamlines, resample_nb, meanpts_resampling=True, out_dtype=np.float32)

    with log_timer("radius_search"):
        l21_radius = clustering_threshold * resample_nb
        dist_mtx = radius_search(slines_arr, None, radius=l21_radius, metric="l21", both_dir=True,
                                 resample=resample_nb)

    with log_timer("connected components and centroid calculation"):
        dist_mtx.data = np.abs(dist_mtx.data)
        dist_mtx = dist_mtx.tocsr()
        list_of_indices = connected_components_indices(dist_mtx)
        list_of_mtx = connected_components_split(dist_mtx, list_of_indices)

        centroid_trk = []
        cluster_map = {}

        for idx, (slines_ids, mtx_i) in enumerate(zip(list_of_indices, list_of_mtx)):
            # Compute centroid streamline using Tractosearch group_to_centroid
            center = group_to_centroid(
                slines_arr[slines_ids], mtx_i, return_cov=False)
            cluster_map[len(centroid_trk)] = len(slines_ids)
            centroid_trk.append(center.tolist())

    return centroid_trk, cluster_map, resample_nb


def _build_kdtree_and_find_pairs(streamlines, num_centroids, L, eps):
    """Build a KDTree on all streamline points and find close pairs.

    Flattens all streamline points into a single array, builds a KDTree,
    queries for point pairs within distance ``eps``, and maps them back to
    unique streamline pairs.

    Parameters
    ----------
    streamlines : list of list
        Centroid streamlines (each with L points of 3 coordinates).
    num_centroids : int
        Number of centroid streamlines.
    L : int
        Number of points per streamline.
    eps : float
        Distance threshold for pair proximity.

    Returns
    -------
    close_pairs : set of tuple
        Set of (i, j) pairs where i < j, representing close streamline pairs.
    """
    with log_timer("KDTree construction"):
        # NumPy Point Flattening
        streamlines_arr = np.array(streamlines, dtype=np.float32)
        all_pts = streamlines_arr.reshape(-1, 3)
        pt_to_streamline = np.repeat(
            np.arange(num_centroids, dtype=np.int32), L)

        tree = KDTree(all_pts)

    with log_timer("KDTree query_pairs (r=%.2f)" % eps):
        pairs = tree.query_pairs(r=eps)

    with log_timer("filtering to unique streamline pairs"):
        close_pairs = set()
        for p1, p2 in pairs:
            sl1 = pt_to_streamline[p1]
            sl2 = pt_to_streamline[p2]
            if sl1 != sl2:
                if sl1 > sl2:
                    sl1, sl2 = sl2, sl1
                close_pairs.add((sl1, sl2))

    return close_pairs


def _compute_events(close_pairs, streamlines, eps, alpha, num_centroids, L):
    """Compute connect/disconnect events for all close streamline pairs.

    For each close pair, calls ``find_connect_disconnect_events`` to determine
    where bundles merge and split, then aggregates the results into a
    per-streamline, per-point event structure.

    Parameters
    ----------
    close_pairs : set of tuple
        Set of (i, j) close streamline pairs.
    streamlines : list of list
        Centroid streamlines.
    eps : float
        Distance threshold for pair proximity.
    alpha : float
        Persistence threshold.
    num_centroids : int
        Number of centroid streamlines.
    L : int
        Number of points per streamline.

    Returns
    -------
    events_T : list of list of list of Event
        events_T[i][t] is the list of events for streamline i at point t.
    next_event_t : numpy.ndarray
        next_event_t[i] is the index of the first point with events for streamline i.
    """
    # Pre-allocate event structures for Direct O(1) index lookups
    events_T = [[[] for _ in range(L)] for _ in range(num_centroids)]
    for i in range(num_centroids):
        events_T[i][0].append(Event("appear"))
        events_T[i][L - 1].append(Event("disappear"))

    # Wrap the event check loop in tqdm progress bar
    with log_timer("event check loop for %d close pairs" % len(close_pairs)):
        for i, j in tqdm(close_pairs, desc="Computing events", disable=len(close_pairs) < 100):
            events_t1, events_t2 = find_connect_disconnect_events(
                i, j, streamlines[i], streamlines[j], eps, alpha)
            for key, evs in events_t1.items():
                events_T[i][key].extend(evs)
            for key, evs in events_t2.items():
                events_T[j][key].extend(evs)

    # Precompute next_event_t for O(1) event checking
    with log_timer("precomputing next_event_t"):
        next_event_t = np.full(num_centroids, L, dtype=np.int32)
        for i in range(num_centroids):
            for t in range(L):
                if events_T[i][t]:
                    next_event_t[i] = t
                    break

    return events_T, next_event_t


def _sweep_line_simulation(num_centroids, L, events_T, next_event_t):
    """Run the sweep-line simulation to assign cluster IDs.

    Processes appear/connect/disconnect/disappear events along the
    streamline points to track dynamic connected components and assign
    cluster identifiers.

    Parameters
    ----------
    num_centroids : int
        Number of centroid streamlines.
    L : int
        Number of points per streamline.
    events_T : list of list of list of Event
        Per-streamline, per-point event lists.
    next_event_t : numpy.ndarray
        Index of the first event point for each streamline.

    Returns
    -------
    cluster_assignments : numpy.ndarray of shape (num_centroids, L)
        Cluster assignment matrix.
    """
    with log_timer("sweep-line simulation loop"):
        G_pres = nx.Graph()  # G_(k)
        clusters_prev_set = set()
        cluster_assignments = np.full((num_centroids, L), -1, dtype=np.int32)

        cluster_id = -1
        del_nodes_set = set()

        # NumPy array to track current point index for each streamline
        t_indices = np.zeros(num_centroids, dtype=np.int32)

        # Progress bar setup
        total_points = num_centroids * L
        pbar = tqdm(total=total_points, desc="Sweep-line simulation",
                    disable=logger.getEffectiveLevel() > logging.INFO)

        while True:
            incomplete_mask = (t_indices < L)
            if not np.any(incomplete_mask):
                break

            # Assume all incomplete streamlines are active
            active_mask = incomplete_mask.copy()

            has_event_indices = np.where(
                (t_indices == next_event_t) & incomplete_mask)[0]

            # Check blocking conditions only for streamlines that have events at their current point
            for s in has_event_indices:
                t_s = t_indices[s]
                events = events_T[s][t_s]
                for e in events:
                    if e.event in ("connect", "disconnect") and t_indices[e.trajectory] < e.t:
                        active_mask[s] = False
                        break

            # Check for deadlock (all incomplete streamlines blocked)
            if not np.any(active_mask):
                first_incomplete = np.where(incomplete_mask)[0][0]
                active_mask[first_incomplete] = True

            # Process events for active streamlines
            any_advanced = np.any(active_mask)
            if not any_advanced:
                break

            advanced_count = np.sum(active_mask)
            pbar.update(int(advanced_count))

            event_occurred = False
            for s in has_event_indices:
                if active_mask[s]:
                    t_s = t_indices[s]
                    events = events_T[s][t_s]
                    if events:
                        event_occurred = True
                        for e in events:
                            if e.event == "appear":
                                G_pres.add_node(s)
                            elif e.event == "connect":
                                G_pres.add_node(s)
                                G_pres.add_node(e.trajectory)
                                G_pres.add_edge(s, e.trajectory)
                            elif e.event == "disconnect":
                                if G_pres.has_edge(s, e.trajectory):
                                    G_pres.remove_edge(s, e.trajectory)
                                    if s in del_nodes_set and nx.is_isolate(G_pres, s):
                                        G_pres.remove_node(s)
                                        del_nodes_set.remove(s)
                                    if e.trajectory in del_nodes_set and nx.is_isolate(G_pres, e.trajectory):
                                        G_pres.remove_node(e.trajectory)
                                        del_nodes_set.remove(e.trajectory)
                            elif e.event == "disappear":
                                if G_pres.has_node(s):
                                    if nx.is_isolate(G_pres, s):
                                        G_pres.remove_node(s)
                                    else:
                                        del_nodes_set.add(s)

            if event_occurred:
                # Connected components
                clusters_pres = list(nx.connected_components(G_pres))
                clusters_pres_frozenset = [frozenset(c) for c in clusters_pres]

                for cluster_pres, cluster_fs in zip(clusters_pres, clusters_pres_frozenset):
                    if cluster_fs not in clusters_prev_set:
                        cluster_id += 1
                        for cluster_traj in cluster_pres:
                            t_traj = t_indices[cluster_traj]
                            if t_traj < L:
                                cluster_assignments[cluster_traj,
                                                    t_traj] = cluster_id
                clusters_prev_set = set(clusters_pres_frozenset)

            # Increment point indices for active streamlines
            t_indices[active_mask] += 1

            # Update next_event_t for those that advanced
            for s in has_event_indices:
                if active_mask[s]:
                    t_next = t_indices[s]
                    while t_next < L and not events_T[s][t_next]:
                        t_next += 1
                    next_event_t[s] = t_next

        pbar.close()

    with log_timer("filling unassigned cluster IDs"):
        for stream_i in range(num_centroids):
            row = cluster_assignments[stream_i]
            for s_i in range(1, L):
                if row[s_i] == -1:
                    row[s_i] = row[s_i - 1]

    return cluster_assignments


def _filter_bundles(cluster_assignments, num_centroids, L, cluster_map, delta):
    """Apply delta-threshold filtering and propagate valid cluster IDs.

    Removes clusters whose total streamline count is at or below ``delta``,
    marks affected cells as -2, and propagates valid cluster IDs forward
    and backward to fill the gaps.

    Parameters
    ----------
    cluster_assignments : numpy.ndarray of shape (num_centroids, L)
        Cluster assignment matrix.
    num_centroids : int
        Number of centroid streamlines.
    L : int
        Number of points per streamline.
    cluster_map : dict
        Mapping from centroid index to cluster size.
    delta : float
        Minimum bundle streamline count threshold.

    Returns
    -------
    cluster_assignments : numpy.ndarray
        Updated cluster assignment matrix.
    deleted_streamlines : set
        Set of streamline indices that are entirely deleted.
    trajectory_counts : dict
        Mapping from cluster ID to total trajectory count.
    """
    with log_timer("bundle deletion and filtering"):
        trajectory_counts = {}
        delete_cluster = set([])
        for stream_i in range(num_centroids):
            # We can extract the row as a list to find unique clusters preserving order
            unique_cluster = list(dict.fromkeys(
                cluster_assignments[stream_i].tolist()))
            for uc in unique_cluster:
                if trajectory_counts.get(uc):
                    trajectory_counts[uc] += cluster_map[stream_i]
                else:
                    trajectory_counts[uc] = cluster_map[stream_i]

        for (x, y) in trajectory_counts.items():
            if y <= delta:
                delete_cluster.add(x)

        # Vectorized check for deleted bundles
        if delete_cluster:
            mask = np.isin(cluster_assignments, list(delete_cluster))
            cluster_assignments[mask] = -2

        # Check for rows where all values are -2
        all_minus_two = np.all(cluster_assignments == -2, axis=1)
        del_s_id = np.where(all_minus_two)[0].tolist()
        deleted_streamlines = set(del_s_id)

        # Forward and backward propagation of valid cluster IDs in-place
        for stream_i in range(num_centroids):
            if stream_i not in deleted_streamlines:
                row = cluster_assignments[stream_i]
                for s_i in range(1, L):
                    if row[s_i] == -2:
                        row[s_i] = row[s_i - 1]
                for s_i in range(L - 2, -1, -1):
                    if row[s_i] == -2:
                        row[s_i] = row[s_i + 1]

    return cluster_assignments, deleted_streamlines, trajectory_counts


def _construct_reeb_graph(cluster_assignments, num_centroids, L, deleted_streamlines, trajectory_counts, cluster_map, streamlines):
    """Build the initial Reeb graph nodes and edges from cluster assignments.

    Creates edges between cluster-transition nodes and computes initial
    node locations from streamline points at cluster boundaries.

    Parameters
    ----------
    cluster_assignments : numpy.ndarray of shape (num_centroids, L)
        Cluster assignment matrix.
    num_centroids : int
        Number of centroid streamlines.
    L : int
        Number of points per streamline.
    deleted_streamlines : set
        Streamline indices that are entirely deleted.
    trajectory_counts : dict
        Mapping from cluster ID to total trajectory count.
    cluster_map : dict
        Mapping from centroid index to cluster size.
    streamlines : list of list
        Centroid streamlines.

    Returns
    -------
    R : networkx.Graph
        Initial Reeb graph (before spatial contractions).
    node_loc : dict
        Mapping from node ID to list of 3D points.
    cluster_edge : dict
        Mapping from cluster ID to [start_node, end_node].
    """
    with log_timer("Reeb Graph nodes/edges construction"):
        R = nx.Graph()
        G_nodes = nx.Graph()
        cluster_edge = {}
        node_loc = {}
        node_id = 0

        for stream_i in range(num_centroids):
            if stream_i not in deleted_streamlines:
                unique_cluster = list(dict.fromkeys(
                    cluster_assignments[stream_i].tolist()))
                if len(unique_cluster) == 1:
                    if not cluster_edge.get(unique_cluster[0]):
                        R.add_edge(node_id, node_id + 1)
                        R[node_id][node_id + 1]['weight'] = trajectory_counts[unique_cluster[0]
                                                                              ] / sum(cluster_map.values())
                        cluster_edge[unique_cluster[0]] = [
                            node_id, node_id + 1]
                        node_id += 2
                for uc in range(len(unique_cluster) - 1):
                    if not cluster_edge.get(unique_cluster[uc]):
                        R.add_edge(node_id, node_id + 1)
                        R[node_id][node_id + 1]['weight'] = trajectory_counts[unique_cluster[uc]
                                                                              ] / sum(cluster_map.values())
                        cluster_edge[unique_cluster[uc]] = [
                            node_id, node_id + 1]
                        node_id += 2
                    if not cluster_edge.get(unique_cluster[uc + 1]):
                        R.add_edge(node_id, node_id + 1)
                        R[node_id][node_id + 1]['weight'] = trajectory_counts[unique_cluster[uc + 1]
                                                                              ] / sum(cluster_map.values())
                        cluster_edge[unique_cluster[uc + 1]
                                     ] = [node_id, node_id + 1]
                        node_id += 2

    # Node location
    with log_timer("computing initial node coordinates from bundle points"):
        for stream_i in range(num_centroids):
            if stream_i not in deleted_streamlines:
                x = cluster_assignments[stream_i, L - 1]
                if cluster_edge[x][1] in node_loc.keys():
                    node_loc[cluster_edge[x][1]].append(
                        streamlines[stream_i][L - 1])
                else:
                    node_loc[cluster_edge[x][1]] = [
                        streamlines[stream_i][L - 1]]
                y = cluster_assignments[stream_i, 0]
                if cluster_edge[y][0] in node_loc.keys():
                    node_loc[cluster_edge[y][0]].append(
                        streamlines[stream_i][0])
                else:
                    node_loc[cluster_edge[y][0]] = [streamlines[stream_i][0]]

                begin = y
                for s_i in range(1, L):
                    if cluster_assignments[stream_i, s_i] != begin:
                        if cluster_edge[begin][1] in node_loc.keys():
                            node_loc[cluster_edge[begin][1]].append(
                                streamlines[stream_i][s_i - 1])
                        else:
                            node_loc[cluster_edge[begin][1]] = [
                                streamlines[stream_i][s_i - 1]]
                        begin = cluster_assignments[stream_i, s_i]
                        if cluster_edge[begin][0] in node_loc.keys():
                            node_loc[cluster_edge[begin][0]].append(
                                streamlines[stream_i][s_i])
                        else:
                            node_loc[cluster_edge[begin][0]] = [
                                streamlines[stream_i][s_i]]

    return R, node_loc, cluster_edge


def _contract_nodes(R, node_loc, cluster_edge, cluster_assignments, num_centroids, L, deleted_streamlines, alpha, use_real_positions):
    """Perform spatial node contractions on the Reeb graph.

    Merges nodes that are spatially close (within ``alpha``) by computing
    centroid positions, relabelling nodes, and removing self-loops and
    isolates from the final graph.

    Parameters
    ----------
    R : networkx.Graph
        Initial Reeb graph.
    node_loc : dict
        Mapping from node ID to list of 3D points.
    cluster_edge : dict
        Mapping from cluster ID to [start_node, end_node].
    cluster_assignments : numpy.ndarray of shape (num_centroids, L)
        Cluster assignment matrix.
    num_centroids : int
        Number of centroid streamlines.
    L : int
        Number of points per streamline.
    deleted_streamlines : set
        Streamline indices that are entirely deleted.
    alpha : float
        Edge length merge threshold.
    use_real_positions : bool
        If True, snap node positions to the closest real streamline point.

    Returns
    -------
    R : networkx.Graph
        Final contracted Reeb graph.
    node_loc_final : dict
        Mapping from node ID to 3D coordinate list.
    """
    with log_timer("spatial node contractions (alpha contraction)"):
        node_loc_final = {}
        node_id = max(R.nodes) + 1 if len(R.nodes) > 0 else 0
        for node_key in node_loc.keys():
            n_x = 0
            n_y = 0
            n_z = 0
            for nk in node_loc[node_key]:
                n_x += nk[0]
                n_y += nk[1]
                n_z += nk[2]
            centroid = [
                n_x / len(node_loc[node_key]), n_y / len(node_loc[node_key]), n_z / len(node_loc[node_key])]
            if use_real_positions:
                node_loc_final[node_key] = find_closest_real_point(
                    centroid, node_loc[node_key])
            else:
                node_loc_final[node_key] = centroid

        G_nodes = nx.Graph()
        for stream_i in range(num_centroids):
            if stream_i not in deleted_streamlines:
                unique_cluster = list(dict.fromkeys(
                    cluster_assignments[stream_i].tolist()))
                for uc in range(len(unique_cluster) - 1):
                    dist1 = squared_distance(node_loc_final[cluster_edge[unique_cluster[uc]][1]],
                                             node_loc_final[cluster_edge[unique_cluster[uc + 1]][0]])
                    dist2 = squared_distance(node_loc_final[cluster_edge[unique_cluster[uc]][1]],
                                             node_loc_final[cluster_edge[unique_cluster[uc + 1]][1]])
                    if dist1 < dist2:
                        G_nodes.add_edge(
                            cluster_edge[unique_cluster[uc]][1], cluster_edge[unique_cluster[uc + 1]][0])
                    else:
                        G_nodes.add_edge(
                            cluster_edge[unique_cluster[uc]][1], cluster_edge[unique_cluster[uc + 1]][1])

        merged_nodes = list(nx.connected_components(G_nodes))
        node_map = {}
        for cluster in merged_nodes:
            if len(cluster) > 1:
                for c in cluster:
                    node_map[c] = node_id
                    if node_id in node_loc_final:
                        centroid = [node_loc_final[node_id][0]/2 + node_loc_final[c][0]/2, node_loc_final[node_id]
                                    [1]/2 + node_loc_final[c][1]/2, node_loc_final[node_id][2]/2 + node_loc_final[c][2]/2]
                        if use_real_positions:
                            all_points = []
                            for node_in_cluster in cluster:
                                if node_in_cluster in node_loc:
                                    all_points.extend(
                                        node_loc[node_in_cluster])
                            node_loc_final[node_id] = find_closest_real_point(
                                centroid, all_points) if all_points else centroid
                        else:
                            node_loc_final[node_id] = centroid
                    else:
                        node_loc_final[node_id] = node_loc_final[c]
                node_id += 1

        H = nx.relabel_nodes(R, node_map)
        G_nodes = nx.Graph()
        count_del_edge = 0
        alpha_sq = alpha * alpha
        for (n1, n2) in list(H.edges):
            if squared_distance(node_loc_final[n1], node_loc_final[n2]) < alpha_sq:
                G_nodes.add_edge(n1, n2)
                count_del_edge += 1

        merged_nodes = list(nx.connected_components(G_nodes))
        node_map = {}
        for cluster in merged_nodes:
            if len(cluster) > 1:
                for c in cluster:
                    node_map[c] = node_id
                    if not node_id in node_loc_final:
                        node_loc_final[node_id] = node_loc_final[c]
                    else:
                        centroid = [node_loc_final[node_id][0]/2 + node_loc_final[c][0]/2, node_loc_final[node_id]
                                    [1]/2 + node_loc_final[c][1]/2, node_loc_final[node_id][2]/2 + node_loc_final[c][2]/2]
                        if use_real_positions:
                            node_loc_final[node_id] = find_closest_real_point(
                                centroid, [node_loc_final[node_id], node_loc_final[c]])
                        else:
                            node_loc_final[node_id] = centroid
                node_id += 1

        R = nx.relabel_nodes(H, node_map)
        R.remove_edges_from(nx.selfloop_edges(R))
        R.remove_nodes_from(list(nx.isolates(R)))

    return R, node_loc_final


def construct_robust_reeb(
    streamlines: List[np.ndarray],
    eps: float,
    alpha: float,
    delta: float,
    clustering_threshold: float = 2.5,
    resample_nb: int = 40,
    use_real_positions: bool = False
) -> Tuple[nx.Graph, Dict[int, List[float]]]:
    """
    Reeb Graph Computation using Tractosearch for initial clustering/downsampling.

    Parameters
    ----------
    streamlines : list of numpy arrays
        Streamlines representing the white matter tracts.
    eps : float
        Distance between a pair of streamlines defining sparsity.
    alpha : float
        Spatial length of the bundle introducing persistence (also edge length merge threshold).
    delta : float
        Minimum bundle streamline count threshold.
    clustering_threshold : float
        Distance threshold for Tractosearch clustering (MDF distance).
    resample_nb : int
        Number of points to resample each streamline to.

    Returns
    -------
    R : networkx.Graph
        Constructed Reeb Graph.
    node_loc_final : dict
        Mapping from node ID to 3D coordinate list.
    """
    t_start = time.perf_counter()
    logger.debug("[Timer] construct_robust_reeb started.")

    if len(streamlines) == 0:
        logger.debug(
            "[Timer] Finished immediately because streamlines are empty.")
        return nx.Graph(), {}

    # 1. Downsampling streamlines using Tractosearch instead of QuickBundles
    centroid_trk, cluster_map, resample_nb = _downsample_streamlines(
        streamlines, clustering_threshold, resample_nb)

    streamlines = centroid_trk
    num_centroids = len(streamlines)
    L = resample_nb

    if not logger.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info(
        "Step 1/6: Downsampling/clustering complete. Centroids count: %d", num_centroids)

    # 2. Sweep-line simulation initialization
    logger.info(
        "Step 2/6: Building KDTree to find physically close streamline pairs...")
    close_pairs = _build_kdtree_and_find_pairs(
        streamlines, num_centroids, L, eps)
    logger.info("Step 3/6: Found %d close interacting pairs out of %d total combinations.",
                len(close_pairs), num_centroids * (num_centroids - 1) // 2)

    # 3. Compute events
    events_T, next_event_t = _compute_events(
        close_pairs, streamlines, eps, alpha, num_centroids, L)

    # 4. Sweep-line simulation
    logger.info(
        "Step 4/6: Running sweep-line simulation to trace bundle lifespan...")
    cluster_assignments = _sweep_line_simulation(
        num_centroids, L, events_T, next_event_t)

    # 5. Filter bundles
    cluster_assignments, deleted_streamlines, trajectory_counts = _filter_bundles(
        cluster_assignments, num_centroids, L, cluster_map, delta)

    # 6. Construct Reeb graph
    logger.info("Step 5/6: Constructing Reeb Graph nodes and edges...")
    R, node_loc, cluster_edge = _construct_reeb_graph(
        cluster_assignments, num_centroids, L, deleted_streamlines, trajectory_counts, cluster_map, streamlines)

    # 7. Spatial node contractions
    logger.info(
        "Step 6/6: Performing spatial node contractions (alpha = %.2f)...", alpha)
    R, node_loc_final = _contract_nodes(
        R, node_loc, cluster_edge, cluster_assignments, num_centroids, L, deleted_streamlines, alpha, use_real_positions)

    logger.debug("[Timer] construct_robust_reeb completed in %.4f seconds.",
                 time.perf_counter() - t_start)
    return R, node_loc_final
