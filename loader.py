import pathlib
import numpy as np
import json
import nibabel as nib

from collections import namedtuple
from typing import Dict, List, Sequence, Tuple, Optional, Callable


DataTriplet = namedtuple('DataTriplet', ('raw', 'label', 'meta'))


def load_nii(rawpath: pathlib.Path, labelpath: pathlib.Path) -> Tuple[np.ndarray]:
    """
    Load the NII image data and return numerical data as plain numpy arrays.
    Additionally, the affine transformation matrix is extracted
    """
    raw = nib.load(rawpath)
    label = nib.load(labelpath).get_fdata()
    return (raw.get_fdata(), label, raw.affine)


def get_numerical_load_fn(filetype: str) -> Callable:
    """Get the load function callable that handles the given filetype."""
    if filetype == 'nii':
        return load_nii
    else:
        raise NotImplementedError('not yet, son!')



def load_metadata(metadatapath: pathlib.Path) -> Dict:
    """Load JSON metadata."""
    with open(metadatapath, mode='r') as handle:
        metadata = json.load(handle)
    return metadata



def load(rawpath: pathlib.Path, labelpath: pathlib.Path,
         metadatapath: pathlib.Path, filetype='nii') -> namedtuple:
    """
    Load a single dataset from the raw, label, metadata
    path triplet.
    """
    # load numerical data
    load_fn = get_numerical_load_fn(filetype)
    (raw, label, affine) = load_fn(rawpath=rawpath, labelpath=labelpath)
    # load metadata
    meta_data = load_metadata(metadatapath=metadatapath)
    # add affine transformation matrix to metadata
    meta_data['COORD']['AFFINE'] = affine
    return DataTriplet(raw=raw, label=label, meta=meta_data)



if __name__ == '__main__':

    from crawler import get_paths

    print('Hello world!')
    res = get_paths('C:/Users/Jannik/Desktop/segmentation_net/seg_net/misc/dataset_wimmer')

    data = load(**res['F01']['CT'])
    
    for elem in data:

        if isinstance(elem, np.ndarray):
            print(elem.dtype)
            print(elem.shape)
        else:
            print(elem)

    print('Goodbye!')

    