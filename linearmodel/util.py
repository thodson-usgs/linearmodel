import copy

import numpy as np
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

    def deepcopy(self):

        return copy.deepcopy(self)


class HDFio:
    """Class saving and retrieving an object state to and from an HDF file"""

    _scalar_types = (str, bool, type(None))
    _list_types = (list, tuple, pd.DatetimeIndex)

    @staticmethod
    def _dict_from_hdf(store, key):
        """

        :param store:
        :param key:
        :return:
        """
        dict_series = pd.read_hdf(store, key)
        return dict(dict_series)

    @staticmethod
    def _dict_to_hdf(store, value, key):
        """

        :param store:
        :param key:
        :return:
        """
        dict_series = pd.Series(data=value)
        dict_series.to_hdf(store, key)

    @staticmethod
    def _list_from_hdf(store, key):
        """

        :param store:
        :param key:
        :return:
        """
        list_series = pd.read_hdf(store, key)
        return list(list_series)

    @staticmethod
    def _list_to_hdf(store, value, key):
        """Write a list to an HDFStore instance

        :param store:
        :param value:
        :param key:
        :return:
        """
        list_series = pd.Series(data=value)
        list_series.to_hdf(store, key)

    @staticmethod
    def _scalar_from_hdf(store, key):
        """

        :param store:
        :param key:
        :return:
        """
        scalar_series = pd.read_hdf(store, key)
        return list(scalar_series)[0]

    @staticmethod
    def _scalar_to_hdf(store, value, key):
        """Write

        :param store:
        :param value:
        :param key:
        :return:
        """
        scalar_series = pd.Series(data=value)
        scalar_series.to_hdf(store, key)

    @classmethod
    def read_hdf(cls, store, attribute_types, key):
        """

        :param store:
        :param attribute_types:
        :param key:
        :return:
        """
        attributes = {}
        for k, value_type in attribute_types.items():
            next_key = key + '/' + k
            if hasattr(value_type, 'read_hdf'):
                attributes[k] = value_type.read_hdf(store, next_key)
            elif value_type in cls._scalar_types:
                attributes[k] = value_type(cls._scalar_from_hdf(store, next_key))
            elif value_type in cls._list_types:
                attributes[k] = value_type(cls._list_from_hdf(store, next_key))
            elif value_type is np.ndarray:
                attributes[k] = np.array(cls._list_from_hdf(store, next_key))
            elif value_type is dict:
                attributes[k] = cls._dict_from_hdf(store, next_key)
            else:
                raise TypeError("Unable to handle type {}".format(value_type))

        return attributes

    @classmethod
    def to_hdf(cls, store, attributes_dict, key):
        """Write contents of attributes_dict to an HDFStore to a path beginning with key.

        :param store: Opened HDFStore
        :param attributes_dict: Dictionary containing attributes to write to HDFStore. The key of the dictionary is
                                appended to the key parameter passed ot this method.
        :param key: Key corresponding to group in store
        :return:
        """

        for k, v in attributes_dict.items():
            next_key = key + '/' + k
            if hasattr(v, 'to_hdf'):
                v.to_hdf(store, next_key)
            elif isinstance(v, cls._scalar_types):
                cls._scalar_to_hdf(store, v, next_key)
            elif isinstance(v, cls._list_types):
                cls._list_to_hdf(store, v, next_key)
            elif isinstance(v, dict):
                cls._dict_to_hdf(store, v, next_key)
            elif isinstance(v, np.ndarray):
                cls._list_to_hdf(store, v, next_key)
            else:
                raise TypeError("Unable to handle type {}".format(v.__class__.__name__))
