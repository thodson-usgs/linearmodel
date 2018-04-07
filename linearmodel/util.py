import copy

import h5py
import pandas as pd


class CopyMixin:
    """Mixin that provides general methods for copy() and deepcopy()"""

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v, in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result


class HDFMixin:
    """Mixin class to provide a general method for saving an object state to an HDF file"""

    def _get_to_hdf_dict(self, hdf_members):
        """Create a dictionary to pass to _to_hdf(). Items in the dictionary are defined in _hdf_members

        :param hdf_members: List of members to save to HDF.
        :return:
        """

        # return a dictionary with key, value pairs, with keys defined in self._hdf_members
        return {key: self.__dict__[key] for key in hdf_members}

    @classmethod
    def _read_hdf(cls, hdf_path, hdf_key='/'):
        """

        :param hdf_path:
        :param hdf_key:
        :return:
        """

        result = cls.__new__(cls)
        with h5py.File(hdf_path, 'r') as f:
            members = list(f[hdf_key])
        for m in members:
            member_key = hdf_key + '/' + m
            df = pd.read_hdf(hdf_path, key=member_key)
            setattr(result, m, df)

        return result

    @classmethod
    def _to_hdf(cls, item_dict, hdf_path, hdf_key):
        """Writes data from item_dict to an HDF file

        :param item_dict: Dictionary containing items to write to HDF
        :param hdf_path: Path to an HDF file
        :param hdf_key: Identifier for the top-level group in the HDF file
        :return: None
        """

        for key, value in item_dict.items():
            # create the next-level key
            next_hdf_key = hdf_key + '/' + key

            # if the item is a dictionary, call this method to write it
            if isinstance(value, dict):
                cls._to_hdf(value, hdf_path, next_hdf_key)

            # otherwise use the item's to_hdf method if it has one or write the item to the HDF
            else:
                try:
                    value.to_hdf(hdf_path, next_hdf_key)
                except AttributeError:
                    print("Saving " + key + " to " + hdf_path)
                    with h5py.File(hdf_path, 'r+') as f:
                        f[next_hdf_key] = value

