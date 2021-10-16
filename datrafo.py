"""
Transform numerical data: Interpolate voxel arrays to
isotropic voxel dimensions.
"""

import numpy as np
import scipy.interpolate

from typing import (Sequence, Dict, List, Tuple,
                    Union, Optional, Any)


class Resampler:

    valid_backends = ['scipy', 'torch']

    def __init__(self,
                 original_voxel_size: Sequence[float],
                 resampled_voxel_size: Sequence[float],
                 backend: str = 'scipy',
                 mode: str = 'linear',
                 device: str = 'cuda') -> None:
        
        self.original_voxel_size = np.array(original_voxel_size)
        self.resampled_voxel_size = np.array(resampled_voxel_size)
        # will be set by backend setter
        self._interpolate_fn = None
        self.backend = backend
        self.mode = mode
        self.device = device


    @property
    def backend(self) -> str:
        return self._backend


    @backend.setter
    def backend(self, b: str) -> None:
        if b not in self.valid_backends:
            raise ValueError(f'Invalid backend {b}! Must be one of {self.valid_backends}')
        self._backend = b
        self._set_interpolate_fn(backend=b)


    def __call__(self, rawdata: np.ndarray) -> np.ndarray:
        return self._interpolate_fn(rawdata, self.mode)


    def _set_interpolate_fn(self, backend: str) -> None:
        if np.allclose(self.original_voxel_size, self.resampled_voxel_size):
            self._interpolate_fn = self._identity_fn
        elif backend == 'scipy':
            self._interpolate_fn = self._interpolate_scipy
        elif backend == 'torch':
            self._interpolate_fn = self._interpolate_torch
        else:
            raise RuntimeError('hic sunt dracones')

    
    def _interpolate_scipy(self, rawdata: np.ndarray, mode: str) -> np.ndarray:
        """
        Interpolate via scipy
        """
        # physical edge sizes of the voxel array  
        physical_axis_sizes = np.array(rawdata.shape) * self.original_voxel_size
        # target shape of the resampled array
        resampled_shape_target = np.rint(
            physical_axis_sizes / self.resampled_voxel_size
        ).astype(np.int32)

        # compute voxel center coordinates
        original_coords = [
            voxel_edge_length * (0.5 + np.arange(axis_size))
            for voxel_edge_length, axis_size in zip(self.original_voxel_size, rawdata.shape)
        ]
        # compute coordinates at which the interpolated voxels are sampled
        resampling_coords = [
            voxel_edge_length * (0.5 + np.arange(axis_size))
            for voxel_edge_length, axis_size in zip(self.resampled_voxel_size, resampled_shape_target)
        ]
        # possible edge mode padding to allow interpolation 'outside' of rawdata domain
        padded_original_coords = []
        pad_widths = []
        for (o_axis_coords, r_axis_coords, edge_length) in zip(original_coords,
                                                               resampling_coords,
                                                               self.original_voxel_size):
            pad_width = self.determine_pad(o_axis_coords, r_axis_coords)
            padded_original_coords.append(
                self.pad_coordinates(o_axis_coords, edge_length, pad_width)
            )
            pad_widths.append(pad_width)
        
        padded_rawdata = self.pad_rawdata(rawdata, pad_widths)

        interpolator = scipy.interpolate.RegularGridInterpolator(
            points=padded_original_coords, values=padded_rawdata
        )

        resampling_grid = np.meshgrid(*resampling_coords, indexing='ij')
        # grid points where the original data is interpolated as a full
        # array of point coordinates
        resampled_grid_as_pcoords = np.concatenate(
            [arr.reshape(-1, 1) for arr in resampling_grid],
            axis=1
        )
        interpolation_result = interpolator(resampled_grid_as_pcoords)
        return np.reshape(interpolation_result, newshape=resampled_shape_target, order='C')


    def _interpolate_torch(self, rawdata: np.ndarray, mode: str) -> np.ndarray:
        raise NotImplementedError('not yet son :O')


    @staticmethod
    def determine_pad(original_coords: np.ndarray, resampling_coords: np.ndarray) -> Tuple[int]:
        """
        Determine padding necessity via coordinate comparison.
        """
        pre_pad = 1 if original_coords[0] > resampling_coords[0] else 0
        post_pad = 1 if original_coords[-1] < resampling_coords[-1] else 0
        return (pre_pad, post_pad)


    @staticmethod
    def pad_coordinates(coordinates: np.ndarray,
                        voxel_edge_length: float,
                        pad_width: Tuple[int]) -> np.ndarray:
        """
        Pad a coordinate array by adding new coordinates based on the given voxel size.
        -> pad_width must be a per-axis (pre-pad, post_pad) specification. 
        """
        (pre_pad, post_pad) = pad_width
        pre_pad_values = np.array(
            [-i * voxel_edge_length + coordinates[0] for i in range(1, pre_pad + 1)]
        )
        post_pad_values = np.array(
            [i * voxel_edge_length + coordinates[-1] for i in range(1, post_pad + 1)]
        )
        return np.concatenate((pre_pad_values, coordinates, post_pad_values))


    @staticmethod
    def pad_rawdata(rawdata: np.ndarray, pad_width: Sequence[int]) -> np.ndarray:
        return np.pad(rawdata, pad_width=pad_width, mode='edge')


    @staticmethod
    def _identity_fn(rawdata: Any, *args, **kwargs) -> Any:
        return rawdata



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    print(scipy.__version__)

    resampler2D = Resampler(
        original_voxel_size=(0.1, 0.1), # 0.1),
        resampled_voxel_size=(0.05, 0.05), # 0.05)
    )

    resampler3D = Resampler(
        original_voxel_size=(0.1, 0.1, 0.1),
        resampled_voxel_size=(0.05, 0.05, 0.05)
    )


    rawdata2D = np.random.default_rng().integers(0, 10, size=(10, 10))
    rawdata3D = np.stack([rawdata2D for _ in range(10)], axis=0)

    resampled_data_2D = resampler2D(rawdata2D)
    resampled_data_3D = resampler3D(rawdata3D)


    print(f'Resampled data shape: {resampled_data_2D.shape}')

    fig, axes = plt.subplots(ncols=2, nrows=2)
    axes = axes.flatten()

    SLICE = np.s_[:]

    ax = axes[0]
    ax.set_title(f'original_data @ slice {SLICE}')
    img1 = ax.matshow(rawdata2D[SLICE, ...])

    vmin, vmax = rawdata2D[SLICE, ...].min(), rawdata2D[SLICE, ...].max()

    ax = axes[1]
    ax.set_title(f'resampled_data @ slice {SLICE}')
    img2 = ax.matshow(resampled_data_2D[SLICE, ...], vmin=vmin, vmax=vmax)

    fig.colorbar(img2, ax=ax)


    print(f'Resampled data shape: {resampled_data_3D.shape}')



    SLICE = 5
    print(f'Slice allclose to 2D: {np.allclose(resampled_data_3D[SLICE, ...], resampled_data_2D)}')

    ax = axes[2]
    ax.set_title(f'original_data @ slice {SLICE}')
    img1 = ax.matshow(rawdata3D[SLICE, ...])

    vmin, vmax = rawdata3D[SLICE, ...].min(), rawdata3D[SLICE, ...].max()

    ax = axes[3]
    ax.set_title(f'resampled_data_3D @ slice {SLICE}')
    img2 = ax.matshow(resampled_data_3D[SLICE, ...], vmin=vmin, vmax=vmax)

    fig.colorbar(img2, ax=ax)

    plt.show()


