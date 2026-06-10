import nibabel as nib


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
