# `reeb_core` - Unified Reeb Graph & Tractosearch Framework

This repository provides a unified framework for topological modeling of white matter pathways (streamlines) as Reeb Graphs. It integrates `Tractosearch` for fast, low-threshold streamline clustering and provides a VTK-based CLI visualizer with synchronized dual cameras.

---

## Features
*   **Unified Core Package**: Consolidated codebase merging Reeb graph construction, proximity event checking, and Siminet distance scoring.
*   **Tractosearch Integration**: Replaces legacy `QuickBundles` clustering with `Tractosearch`'s fast $L_{2,1}$ Mixed-Norm KDTree search.
*   **VTK CLI Visualizer**: Command-line interface showing side-by-side viewports of physical streamlines and the abstracted Reeb graph, with synchronized camera movements.

---

## Installation

First, ensure that the local dependency `tractosearch` is installed. From the parent directory:
```bash
pip install -e ./tractosearch
```

Then, install `reeb_core` in editable mode:
```bash
pip install -e ./reeb_core
```

---

## CLI Visualizer Usage

You can launch the visualizer using the registered command-line entry point:
```bash
reeb-visualize <path_to_tractogram.trk> --epsilon 1.5 --alpha 2.0 --delta 2
```

### Options
*   `in_tractogram`: Path to the input `.trk` or `.tck` file.
*   `--epsilon` / `-e`: Spatial connection proximity threshold (default: 1.5).
*   `--alpha` / `-a`: Spatial persistence length threshold (default: 2.0).
*   `--delta` / `-d`: Minimum bundle streamline count threshold (default: 2).

---

## Optimizations

The core algorithm (`construct_robust_reeb`) was profiled and refactored to eliminate
several performance bottlenecks from the original student implementations.
The changes are purely algorithmic — the topological result is identical.

### 1. Sweep-line simulation: O(N) → O(events)

The sweep-line loop is the heart of the algorithm.
It advances a cursor along every streamline in lock-step and processes topological
events (bundles connecting or disconnecting) at each step.
Three independent bottlenecks were identified and fixed:

**a) Event scanning — O(N) per tick → O(1)**

The original code iterated over *every* streamline on *every* tick to check whether
it had an event waiting at the current point.

```python
# Before — O(N) scan every tick
for s in range(num_centroids):
    if events_T[s][t_indices[s]]:
        ...
```

The fix precomputes `next_event_t[i]` — the index of the next point that has an event
for streamline `i`.  A single vectorised comparison then finds only the streamlines
that actually need attention:

```python
# After — O(1) lookup
has_event_indices = np.where((t_indices == next_event_t) & incomplete_mask)[0]
```

**b) Disappeared-node tracking — O(|del_nodes|) per tick → O(1) per event**

When a streamline "disappears" it was added to a `del_nodes` list.
Every tick, the loop scanned the *entire* list calling `nx.is_isolate()` on each entry.
As the list grew the loop became progressively slower.

The fix uses a `del_nodes_set` and triggers the isolation check only at the precise
moment an edge is removed (during a `disconnect` event) — the only moment when a node
*can* become isolated:

```python
elif e.event == "disconnect":
    if G_pres.has_edge(s, e.trajectory):
        G_pres.remove_edge(s, e.trajectory)
        # Check isolation only here, not on every tick
        if s in del_nodes_set and nx.is_isolate(G_pres, s):
            G_pres.remove_node(s)
            del_nodes_set.remove(s)
```

**c) Cluster-ID forward-fill — O(N·L) per tick → one linear pass**

Inside the loop, unassigned time-steps (value `-1`) were being filled in on every
inactive tick, causing redundant work proportional to the full matrix size.
The fill was moved entirely outside the loop as a single post-processing pass:

```python
# After the loop — run once, O(N·L) total
for stream_i in range(num_centroids):
    row = cluster_assignments[stream_i]
    for s_i in range(1, L):
        if row[s_i] == -1:
            row[s_i] = row[s_i - 1]
```

---

### 2. Numba JIT on distance hot-paths

The event-detection step (`find_connect_disconnect_events` in `events.py`) calls
distance checks across *every point* of every close streamline pair.
With hundreds of pairs and 40 points each, this executes millions of 3-D distance
comparisons per tractogram.

The two innermost numeric functions were compiled with Numba:

```python
from numba import njit

@njit(cache=True)
def check_epsilon_distance(p1, p2, eps):
    ...

@njit(cache=True)
def _try_expand_direction(streamline_i, streamline_j, t, direction, eps, alpha):
    ...
```

`cache=True` means compilation happens once on the first call; subsequent runs load
the cached binary directly.
The sweep-line loop itself cannot use Numba because it depends on `networkx.Graph`
for live connected-component tracking, which Numba's `nopython` mode does not support.

---

### 3. Dead code removal & KDTree hardening (`distance.py`)

- Removed the unused `avg_del_dist` function that was never called.
- Hardened the centrality computation against isolated-node and empty-graph edge cases.

---

### 4. Decomposition & timing instrumentation

The original ~600-line monolithic `constructRobustReeb` was split into **7 focused
helper functions** (`_downsample_streamlines`, `_build_kdtree_and_find_pairs`,
`_compute_events`, `_sweep_line_simulation`, `_filter_bundles`,
`_construct_reeb_graph`, `_contract_nodes`).

Scattered `time.perf_counter()` pairs were replaced with a `log_timer()` context
manager (see `reeb_core/utils/timing.py`).
Set the `reeb_core` logger to `DEBUG` to see per-step timing; set it to `WARNING` to
silence it entirely.

---

## References & Disclaimer

> **Note:** This repository is a **personal experimentation project** aimed at speeding up existing code and adapting it to a new pipeline. It is not an original contribution. All core ideas, algorithms, and implementations originate from the works listed below.

### Scientific Articles

1. **S. Shailja, J. W. Chen, S. T. Grafton, and B. S. Manjunath** (2023).
   *ReTrace: Topological evaluation of white matter tractography algorithms using Reeb graphs.*
   bioRxiv preprint. DOI: [10.1101/2023.07.03.547451](https://doi.org/10.1101/2023.07.03.547451)

2. **S. Shailja, V. Bhagavatula, M. Cieslak, J. M. Vettel, S. T. Grafton, and B. S. Manjunath** (2023).
   *ReeBundle: A Method for Topological Modeling of White Matter Pathways Using Diffusion MRI.*
   *IEEE Transactions on Medical Imaging.* PMID: [37590108](https://pubmed.ncbi.nlm.nih.gov/37590108/)

### Source Code

- **ReTrace** — S. Shailja et al.:
  [https://github.com/s-shailja/ReTrace](https://github.com/s-shailja/ReTrace)

- **ReeBundle** — UCSB Vision Research Lab:
  [https://github.com/UCSB-VRL/ReeBundle](https://github.com/UCSB-VRL/ReeBundle)
