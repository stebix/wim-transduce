"""
Transduce Wimmer datasets into the custom HDF5 landmarked dataset,
"""
import pathlib
import argparse
import h5py
import tqdm

from typing import (Union, Dict, Sequence,
                    Optional)
from crawler import get_paths
from loader import load
from lmtrafo import create_markup_dicts
from datrafo import DummyResampler, Resampler


PathLike = Union[str, pathlib.Path]
RAW_STEM = 'raw'
LABEL_STEM = 'label'
LANDMARK_STEM = 'landmark'
INTERNAL_PATHS = ['raw', 'label', 'landmark']
WIMMER_VOXEL_SIZE = (0.15, 0.15, 0.20)
DEFAULT_VOXEL_SIZE = (0.099, 0.099, 0.099)


def main():
    parser = argparse.ArgumentParser(
        prog='transduce.py',
        description=('Transduces Wimmer datasets form Zenodo into the '
                     'custom HDF5 LandmarkedDataset format.'),
        epilog='Created by JS @ 2021'
    )
    parser.add_argument('sourcedir', type=str, help='Source directory for the Wimmer dataset.')
    parser.add_argument('targetdir', type=str,
                        help='Target directory where the HDF5 files are stored.')
    parser.add_argument('--noresample', '-n', action='store_true',
                        help='Skip resampling altogether and merely transduce into HDF5 file.')
    parser.add_argument('--resample', '-r', nargs=3, default=DEFAULT_VOXEL_SIZE,
                        metavar=('width', 'height', 'depth'),
                        help=f'Resample the CT data to the given voxel size. Default: {DEFAULT_VOXEL_SIZE}')
    parser.add_argument('--original_voxel_size', nargs=3, default=WIMMER_VOXEL_SIZE,
                        help=f'Override defaults for Wimmer dataset voxel size. Default: {WIMMER_VOXEL_SIZE}',
                        metavar=('width', 'height', 'depth'))
    parser.add_argument('--modality', type=str, choices=['CT', 'uCT', 'all'], default='all',
                        help='Select only a single modality to be transduced.')
    parser.add_argument('--no_create_dir', action='store_true',
                        help='Raise error if target directory is not existing.')
    parser.add_argument('--force', action='store_true',
                        help='Force overwriting of pre-existing files at the target directory.')
    parser.add_argument('--omit_modality_fname', '-o', action='store_true',
                        help='Omit the integration of the modality into the HDF5 file name.')
    parser.add_argument('--no_pbar', action='store_true', help='Omit progress display via bar.')
    
    args = parser.parse_args()

    # prepare data
    pathmap = sieve_modality(get_paths(args.sourcedir), args.modality)
    print(pathmap)
    original_voxel_size = tuple((float(f) for f in args.original_voxel_size))
    resampled_voxel_size = tuple((float(f) for f in args.resample))
    if args.noresample:
        original_voxel_size = None
        resampled_voxel_size = None

    transduce(pathmap=pathmap, target_dir=args.targetdir,
              resampled_voxel_size=resampled_voxel_size,
              original_voxel_size=original_voxel_size,
              omit_modality_fname=args.omit_modality_fname,
              no_pbar=args.no_pbar, force=args.force)

    return None


def sieve_modality(pathmap: Dict, modality: str) -> Dict:
    """
    Select entries of the pathmap that correspond to the given modality.
    """
    if modality == 'all':
        return pathmap
    sieved_pathmap = {dset_id : {modality : dset[modality]}
                      for dset_id, dset in pathmap.items()}
    return sieved_pathmap
    
    

def transduce(pathmap: Dict, target_dir: PathLike,
              resampled_voxel_size: Optional[Sequence[float]],
              original_voxel_size: Optional[Sequence[float]],
              omit_modality_fname: bool = False, no_pbar: bool = False,
              force: bool = False) -> None:
    """
    Transduce the objects laid out in pathmap from Wimmer directory-styled
    datasets to HDF5-stebix-styled datasets.
    Transformations:
        - Get IJK position of CochleaTop, Oval + Round Window @ HDF5 landmarks attrs
        - Maybe resample spatially 
    """
    target_dir = pathlib.Path(target_dir)
    if resampled_voxel_size is None and original_voxel_size is None:
        resampler = DummyResampler()
        print('Using dummy resampler ...')
    else:
        print('Using actual resampler ...')
        resampler = Resampler(original_voxel_size, resampled_voxel_size)
    pathmap_iter = pathmap.items() if no_pbar else tqdm.tqdm(pathmap.items())
    for dset_id, modalities in pathmap_iter:
        for modality, instance_pathmap in modalities.items():
            # create full target filepath
            mod_fname = f'_{modality}' if not omit_modality_fname else ''
            filename = ''.join((dset_id, mod_fname, '.hdf5'))
            target_path = target_dir / filename
            if target_path.is_file() and not force:
                raise FileExistsError(f'Fatal: Preexisting file @ "{target_path.resolve()}"! Aborting')
            # create HDF5 file for this dataset + modality combination
            _transduce_dataset(instance_pathmap, target_path, resampler)
    return None


def _transduce_dataset(source_path_map: Dict, target_path: pathlib.Path, resampler: Resampler,
                       raw_stem: str = RAW_STEM,
                       label_stem: str = LABEL_STEM,
                       landmark_stem: str = LANDMARK_STEM) -> None:
    """
    Transduce a single consisting of raw data, label data and metadata to a monolithic
    HDF5 file. 
    """
    data = load(**source_path_map)
    markups = create_markup_dicts(metadata_dict=data.meta)
    with h5py.File(target_path, mode='a') as h5_handle:
        # write array data to HDF5 file
        for stem, array in zip((raw_stem, label_stem), (data.raw, data.label)):
            internal_path = '/'.join((stem, f'{stem}-0')) 
            # resampler may be effectless if no resampling is desired
            h5_handle[internal_path] = resampler(array)
        # write landmark metadata
        landmark_group = h5_handle.create_group(name=landmark_stem, track_order=True)
        for i, landmark_dict in enumerate(markups):
            dataset_name = f'{landmark_stem}-{i}' 
            placeholder_dataset = landmark_group.create_dataset(dataset_name, data=h5py.Empty('f'))
            for key, value in landmark_dict.items():
                placeholder_dataset.attrs[key] = value    
    return None







if __name__ == '__main__':
    main()

    raise Exception

    res = get_paths('C:/Users/Jannik/Desktop/segmentation_net/seg_net/misc/dataset_wimmer')
    print(res['F01'])
    print(res['F02'])

    print('\n\n')

    res_sieved = sieve_modality(res, 'CT')
    print(res_sieved['F01'])
    print(res_sieved['F02'])

