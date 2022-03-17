from abc import ABC, abstractmethod
from copy import copy
import numpy as np
from scipy.interpolate import griddata

class Mesher(ABC):
    """Abstract class for meshing the bathymetrical data to the grid."""
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, data, lon, lat, lonQ, latQ):
        """Gets the bathymetrical information and returns a version that is
        meshed to the area and resolution of the grid.

        data, lon, lat = 1D np.arrays of same length (depth >0, other is nan)
        lonQ, latQ = 2D np.arrays of desired grid size

        The returned array should have the dimensions and orientation:

        rows = latitude and colums = longitude
        I.e. shape = ("lat", "lon").

        North = [-1,:]
        South = [0,:]
        East = [:,-1]
        West = [:,0]

        This method is called from within the Grid-object
        """

        return meshed_data

    @abstractmethod
    def __str__(self):
        """Describes how the data is meshed.

        This is called by the Grid-objeect to provide output to the user.
        """
        pass


class Interpolate(Mesher):
    """Interpolates data to grid. A wrapper for scipy.interpolate's griddate."""

    def __init__(self, method: str='linear') -> None:
        self.method = method

        return

    def __call__(self, data, lon, lat, lonQ, latQ):
        #lon0, lat0 = np.meshgrid(lon, lat)
        data[np.logical_not(data>0)] = 0 # Keeping land points as nan lets the shoreline creep out
        #M = np.column_stack((data.ravel(), lon0.ravel(),lat0.ravel()))
        M = np.column_stack((data, lon, lat))
        meshed_data = griddata(M[:,1:], M[:,0], (lonQ, latQ), method=self.method)

        return meshed_data

    def __str__(self):
        return(f"Meshing using {self.method} interpolation.")

class DaskInterpolate(Mesher):
    """Interpolates data to grid. A wrapper for scipy.interpolate."""

    def __init__(self, client) -> None:
        super().__init__()
        self.client = client

        return

    def __call__(self, da, lonQ, latQ):
        from .dask_interpolate import parallel_interpolate
        #lon0, lat0 = np.meshgrid(lon, lat)
        da = da.where(~np.logical_not(da>0), 0.) # Keeping land points as nan lets the shoreline creep out
        meshed_data = parallel_interpolate(self.client, da, lonQ, latQ)#, chunksize_x=700, chunksize_y=700)

        return meshed_data


    def __str__(self):
        return(f"Meshing using dask and linear interpolation.")


# class TrivialMesher(Mesher):
#     """Passes along data.
#
#     NB! This might not fit the grid, and is only used for e.g. recreating a
#     Grid-object from an ouput file.
#     """
#
#     def __init__(self):
#         pass
#
#     def __call__(self, data, lon, lat, lonQ, latQ):
#         return copy(data)
#
#     def __str__(self):
#         return("Passing input data along as final meshed grid.")
