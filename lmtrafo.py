"""
transform landmark information to local standards.
"""
import numpy as np

from copy import copy
from typing import Dict, List, Sequence, Tuple, Optional, Callable


MARKUP_TEMPLATES = {
    'cochlea_top' : {
        'label' : 'CochleaTop',
        'id' : 1,
        'coordsys' : 'NIBABEL_AFFINE',
        'xyz_position' : None,
        'ijk_position' : None,
        'orientation' : 'NIBABEL_ORIENTATION'
    },
    'oval_window' : {
        'label' : 'OvalWindow',
        'id' : 2,
        'coordsys' : 'NIBABEL_AFFINE',
        'xyz_position' : None,
        'ijk_position' : None,
        'orientation' : 'NIBABEL_ORIENTATION'
    },
    'round_window' : {
        'label' : 'RoundWindow',
        'id' : 3,
        'coordsys' : 'NIBABEL_AFFINE',
        'xyz_position' : None,
        'ijk_position' : None,
        'orientation' : 'NIBABEL_ORIENTATION'
    }
}


def expand_to_4D(vector: np.ndarray) -> np.ndarray:
    vector = vector.reshape(-1, 1)
    return np.concatenate([vector, np.array([1])[:, np.newaxis]])


def reduce_from_4D(vector: np.ndarray) -> np.ndarray:
    return np.squeeze(vector)[:-1]


def get_xyz_position(metadata_dict: dict, landmark_label: str) -> np.ndarray:
    """
    Extract the XYZ position from the given metadata dictionary for
    the landmark given by the (Wimmer-internal) landmark label.
    Hint:
     - C, A, OW, V
    """
    return np.array(metadata_dict['LANDMARKS'][landmark_label])


def xyz_to_ijk(Tijk: np.ndarray, xyz_position: np.ndarray) -> np.ndarray:
    """
    Transform the XYZ position vector to the IJK voxel coordinate vector
    via the transformation matrix `Tijk`.
    """
    v_ijk_homogenous = np.rint(Tijk @ expand_to_4D(xyz_position))
    return reduce_from_4D(v_ijk_homogenous.astype(np.int64))


def finalize_markup_dict(landmark_str_id: str, xyz_position: np.ndarray,
                         ijk_position: np.ndarray) -> Dict:
    """
    Create the landmark-specific markup dict from a template by filling in
    the missing information.
    """
    template = copy(MARKUP_TEMPLATES[landmark_str_id])
    template['xyz_position'] = xyz_position
    template['ijk_position'] = ijk_position
    return template


def create_markup_dicts(metadata_dict: Dict) -> List[Dict]:
    """
    Create the list of markup information dictionaries from the metadata dictionary.
    """
    Tijk = np.linalg.inv(metadata_dict['COORD']['AFFINE'])
    # map the wimmer-specific anatomical landmark label names to our
    # own nomencalture 
    wimmer_internal_label_map = {'A' : 'cochlea_top', 'OW' : 'oval_window',
                                 'RW' : 'round_window'}
    
    markup_dicts = []
    
    for wimmer_label, our_label in wimmer_internal_label_map.items():
        xyz_position = get_xyz_position(metadata_dict, wimmer_label)
        ijk_position = xyz_to_ijk(Tijk, xyz_position)
        markup_dicts.append(
            finalize_markup_dict(our_label, xyz_position, ijk_position)
        )
    return sorted(markup_dicts, key=lambda x: x['id'])



if __name__ == '__main__':

    from crawler import get_paths
    from loader import load

    print('Hello world!')
    res = get_paths('C:/Users/Jannik/Desktop/segmentation_net/seg_net/misc/dataset_wimmer')

    data = load(**res['F01']['CT'])

    markups = create_markup_dicts(data.metadata)    

    print('Goodbye!')
