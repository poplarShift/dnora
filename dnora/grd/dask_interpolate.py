"""
This module shows how to chunk and parallelize interpolation of large regularly gridded
data on arbitrary points.

ara@akvaplan.niva.no, March 2022. Released under GPL2.
"""

import sys

import numpy as np
from dask import delayed
import dask.array as darr
from scipy.interpolate import LinearNDInterpolator
import dask
import xarray as xr

def parallel_interpolate(
    client, da, x_target, y_target, target_partition_size=5000000,
    chunksize_x=400, chunksize_y=400,
):
    chunks = list(c for c in chunk_data_overlap(da, chunksize_x, chunksize_y))
    chunk_sample = chunks[0]
    arr_list = []
    i = 0 # counter for x/y_target partitions
    N = len(x_target)
    while i < N:
        print('Chunking : ', i)
        ind = slice(i, min(i+target_partition_size, N))
        x = x_target[ind]
        y = y_target[ind]
        i += target_partition_size

        arr_list_partition = []
        for c in chunks:
            interpolated_delayed = interpolate(c, x, y)
            interpolated_arr = darr.from_delayed(interpolated_delayed, shape=x.shape,
                                                 dtype=chunk_sample.dtype)
            arr_list_partition.append(interpolated_arr)

        arr_list.append(darr.stack(arr_list_partition))

    stacked = darr.concatenate(arr_list, axis=1)
    # return stacked
    task = darr.nanmean(stacked, axis=0)
    # return task
    print('Computing task :', task)
    future = client.compute(task)
    topo = future.result().compute()
    return topo


def chunk_data_overlap(da, ci, cj):
    """
    Make overlapping chunks
    ci, cj: Chunk sizes in x/y direction
    """
    i = 0
    I, J = len(da.lon), len(da.lat)
    while i < I:
        i_lo = max(i-1, 0)
        i_hi = min(i+ci+1, I)
        i += ci
        j = 0
        while j < J:
            j_lo = max(j-1, 0)
            j_hi = min(j+cj+1, J)
            j += cj
            yield da.isel(lon=slice(i_lo, i_hi), lat=slice(j_lo, j_hi))


def array_coords_overlap(coords1, coords2):
    """
    Check whether two sets of coordinates overlap (box-wise).

    Parameters
    ----------
    coords1, coords2 : tuples of arrays
    """
    for coord1, coord2 in zip(coords1, coords2):
        if (
            coord1.min() > coord2.max()
            or coord1.max() < coord2.min()
        ):
            return False
    return True


@delayed
def interpolate(da, x_target, y_target):
    if array_coords_overlap((da.lon.values, da.lat.values), (x_target, y_target)):
        x, y = np.meshgrid(da.lon, da.lat)
        points = np.stack([x.flatten(), y.flatten()]).T
        values = da.values.flatten()
        interpolator = LinearNDInterpolator(points, values)
        return interpolator(x_target, y_target)
    else:
        return np.nan * np.zeros_like(x_target)
