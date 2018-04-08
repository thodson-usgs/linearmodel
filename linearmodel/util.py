import abc
import copy

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

    @classmethod
    def _read_hdf(cls, member_list, store, key):
        """

        :param member_list:
        :param store:
        :param key:
        :return:
        """

        result = cls.__new__(cls)
        for member_name in member_list:
            member_key = key + '/' + member_name
            try:
                member_value = pd.read_hdf(store, member_key)
            except TypeError:
                member_value = cls.read_hdf(store, member_key)
            setattr(result, member_name, member_value)

        return result

    @classmethod
    def _to_hdf(cls, item_dict, store, key):
        """Writes data from item_dict to an HDF file

        :param item_dict: Dictionary containing items to write to HDF
        :param store: Open HDFStore object
        :param key: Identifier for the top-level group in the HDF file
        :return: None
        """
        if not store.is_open:
            raise IOError('The HDFStore must be open')

        for k, v in item_dict.items():
            # create the next-level key
            next_hdf_key = key + '/' + k

            # if the item is a dictionary, call this method to write it
            if isinstance(v, dict):
                cls._to_hdf(v, store, next_hdf_key)

            # otherwise use the item's to_hdf method
            else:
                v.to_hdf(store, next_hdf_key)

    @classmethod
    @abc.abstractmethod
    def read_hdf(cls, path_or_buf, key=None):
        """

        :param path_or_buf:
        :param key:
        :return:
        """
        pass
