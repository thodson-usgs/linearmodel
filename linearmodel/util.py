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
            else:
                raise TypeError("Unable to handle type {}".format(v.__class__.__name__))
