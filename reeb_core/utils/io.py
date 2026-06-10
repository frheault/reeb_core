import nibabel as nib
import json
import networkx as nx


def load_streamlines(filepath):
    """
    Loads streamlines from a .trk or .tck file using nibabel.

    Parameters
    ----------
    filepath : str
        Path to the streamline file.

    Returns
    -------
    streamlines : list of numpy arrays
        List of 3D streamline coordinate arrays.
    header : dict
        File header metadata.
    """
    sft = nib.streamlines.load(filepath)
    return list(sft.streamlines), sft.header


def save_reeb_graph(filepath: str, R: nx.Graph, node_loc: dict, metadata: dict = None):
    """Save a Reeb graph and its metadata to a JSON file."""
    if metadata is None:
        metadata = {}
    
    # Ensure node_loc is purely JSON serializable (lists instead of numpy arrays)
    clean_node_loc = {int(k): (list(v) if hasattr(v, 'tolist') else list(v)) for k, v in node_loc.items()}
    
    data = {
        "metadata": metadata,
        "graph": nx.node_link_data(R),
        "node_loc": clean_node_loc
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_reeb_graph(filepath: str):
    """Load a Reeb graph and its metadata from a JSON file.
    
    Returns
    -------
    R : networkx.Graph
    node_loc : dict
    metadata : dict
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    R = nx.node_link_graph(data['graph'])
    # Reconstruct dictionary with integer keys
    node_loc = {int(k): v for k, v in data['node_loc'].items()}
    metadata = data.get('metadata', {})
    
    return R, node_loc, metadata
