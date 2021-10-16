import pathlib

from typing import Union, List, Tuple, Dict, NewType



PathLike = NewType(name='PathLike', tp=Union[str, pathlib.Path])


def crawl_subdir(directory: pathlib.Path, dataset_name: str) -> Dict:
    pathmap = {}
    for modality in ('CT', 'uCT'):

        pathmap[modality] = {
            'metadatapath' : directory / f'{dataset_name}_DESC.json'
        }
        subdir = directory / modality
        # access the numerical data for the two CT modalities
        assert subdir.is_dir(), f'directory for modality {modality} not found'
        labelpath = subdir / f'{dataset_name}_{modality}_LABELS.nii'
        rawpath = subdir / f'{dataset_name}_{modality}_RAW.nii'
        pathmap[modality]['labelpath'] = labelpath
        pathmap[modality]['rawpath'] = rawpath

    return pathmap
    


def get_paths(directory: PathLike) -> Dict:
    """
    Crawl the given directory for the Wimmer dataset file structure
    and return a map of filepaths.
    """
    directory = pathlib.Path(directory)
    dataset_map = {}
    for candidate in directory.iterdir():
        if not candidate.is_dir():
            continue
        dataset_name = candidate.stem
        dataset_map[dataset_name] = crawl_subdir(candidate, dataset_name)
    return dataset_map

if __name__ == '__main__':
    res = get_paths('C:/Users/Jannik/Desktop/segmentation_net/seg_net/misc/dataset_wimmer')
    print(res['F01'])
    print(res['F02'])

