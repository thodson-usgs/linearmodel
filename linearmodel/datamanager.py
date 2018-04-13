from datetime import timedelta

import pandas as pd
import numpy as np

from linearmodel.util import CopyMixin, HDFio


class DataException(Exception):
    """Base class for all exceptions in the data module"""
    pass


class DataOriginError(DataException):
    """An error if origin information is inconsistent with the data at Data subclass initialization"""
    pass


class ConcurrentObservationError(DataException):
    """An error if concurrent observations exist for a single variable."""
    pass


class DataManager(CopyMixin):
    """Class for data management."""

    _hdf_members = ['_data', '_data_origin']

    def __init__(self, data, data_origin=None):
        """Initialize a Data object.

        data_origin must be a DataFrame that describes the origin of all columns in the data parameter. At least one
        row per variable. The column names of data_origin must be 'variable' and 'origin.' A brief example follows.

            data_origin example:

                 variable             origin
            0           Q   Q_ILR_WY2016.txt
            1           Q   Q_ILR_WY2017.txt
            2          GH   Q_ILR_WY2017.txt
            3   Turbidity   TurbILR.txt

        :param data: Pandas DataFrame with time DatetimeIndex index type.
        :type data: pd.DataFrame
        :param data_origin: Pandas DataFrame containing variable origin information.
        :type data_origin: pd.DataFrame
        """

        if data_origin is None:
            data_origin = self._create_empty_origin(data)

        self._check_origin(data, data_origin)

        self._data = data.copy(deep=True)
        self._data.sort_index(axis=1, inplace=True)
        if isinstance(self._data.index, pd.DatetimeIndex):
            self._data.index.name = 'DateTime'
        self._data_origin = data_origin.copy(deep=True)

    def __eq__(self, other):
        """

        :param other:
        :return:
        """

        return self.equals(other)

    def __ne__(self, other):
        """

        :param other:
        :return:
        """
        return not self.equals(other)

    def _check_for_concurrent_obs(self, other):
        """Check other DataManager for concurrent observations of a variable. Raise ConcurrentObservationError if
        concurrent observations exist.

        :param other:
        :type other: DataManager
        :return:
        """

        # check for concurrent observations between self and other DataManager
        self_variables = self.get_variable_names()

        other_variables = other.get_variable_names()

        for variable in other_variables:

            if variable in self_variables:

                current_variable = self.get_variable(variable)
                new_variable = other.get_variable(variable)

                if np.any(new_variable.index.isin(current_variable.index)):

                    # raise exception if concurrent observations exist
                    raise ConcurrentObservationError("Concurrent observations exist for variable {}".format(variable))

    @staticmethod
    def _check_origin(data, origin):
        """

        :param origin:
        :return:
        """

        if not isinstance(origin, pd.DataFrame):
            raise TypeError("Origin must be type pandas.DataFrame")

        correct_origin_columns = {'variable', 'origin'}

        origin_columns_difference = correct_origin_columns.difference(origin.keys())

        if len(origin_columns_difference) != 0:
            raise DataOriginError("Origin DataFrame does not have the correct column names")

        variables_grouped = origin.groupby('variable')
        origin_variable_set = set(list(variables_grouped.groups))

        data_variable_set = set(list(data.keys()))

        if not (origin_variable_set.intersection(data_variable_set) == origin_variable_set.union(data_variable_set)):
            raise DataOriginError("Origin and data variables do not match")

    @staticmethod
    def _check_timestamp(value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, pd.tslib.Timestamp):
            raise TypeError('Expected type pandas.tslib.Timestamp, received {}'.format(type(value)), value)

    def _check_variable_name(self, variable_name):
        """

        :param variable_name:
        :type variable_name: str
        :return:
        """

        if variable_name not in self.get_variable_names():
            raise ValueError('{} is not a valid variable name'.format(variable_name), variable_name)

    @classmethod
    def _create_empty_origin(cls, data):
        """
        
        :return: 
        """

        origin = cls.create_data_origin(data, [np.NaN])

        return origin

    @staticmethod
    def _load_tab_delimited_data(file_path):
        """

        :param file_path:
        :return:
        """

        # Read TAB-delimited txt file into a DataFrame.
        tab_delimited_df = pd.read_table(file_path, sep='\t')

        # Check the formatting of the date/time columns. If one of the correct formats is used, reformat
        # those date/time columns into a new timestamp column. If none of the correct formats are used,
        # return an invalid file format error to the user.
        date_time_column_present = True
        if 'y' and 'm' and 'd' and 'H' and 'M' and 'S' in tab_delimited_df.columns:
            tab_delimited_df.rename(columns={"y": "year", "m": "month", "d": "day"}, inplace=True)
            tab_delimited_df.rename(columns={"H": "hour", "M": "minute", "S": "second"}, inplace=True)
            tab_delimited_df["year"] = pd.to_datetime(tab_delimited_df[["year", "month", "day", "hour",
                                                                        "minute", "second"]], errors="coerce")
            tab_delimited_df.rename(columns={"year": "DateTime"}, inplace=True)
            tab_delimited_df.drop(["month", "day", "hour", "minute", "second"], axis=1, inplace=True)
        elif 'Date' and 'Time' in tab_delimited_df.columns:
            tab_delimited_df["Date"] = pd.to_datetime(tab_delimited_df["Date"] + " " + tab_delimited_df["Time"],
                                                      errors="coerce")
            tab_delimited_df.rename(columns={"Date": "DateTime"}, inplace=True)
            tab_delimited_df.drop(["Time"], axis=1, inplace=True)
        elif 'DateTime' in tab_delimited_df.columns:
            tab_delimited_df["DateTime"] = pd.to_datetime(tab_delimited_df["DateTime"], errors="coerce")
        else:
            # raise ValueError("Date and time information is incorrectly formatted.", file_path)
            date_time_column_present = False

        if date_time_column_present:
            tab_delimited_df.set_index("DateTime", drop=True, inplace=True)

        tab_delimited_df = tab_delimited_df.apply(pd.to_numeric, args=('coerce', ))

        return tab_delimited_df

    def add_data(self, data_df, origin=None):
        """Add data from a DataFrame

        :param data_df: DataFrame containing observations to add
        :param origin: Origin information for the variables contained in data_df
        :return: None
        """

        if origin is None:
            data_origin = self._create_empty_origin(data_df)
        else:
            data_origin = self.create_data_origin(data_df, origin)

        other_data_manager = type(self)(data_df, data_origin)

        return self.add_data_manager(other_data_manager)

    def add_data_manager(self, other, keep_curr_obs=None):
        """Add data from other DataManager subclass.

        This method adds the data and data origin information from other DataManager objects. An exception will be
        raised if keep_curr_obs=None and concurrent observations exist for variables.

        :param other: Other DataManager object.
        :type other: DataManager
        :param keep_curr_obs: Indicate whether or not to keep the current observations.
        :type keep_curr_obs: {None, True, False}
        :return:
        """

        if keep_curr_obs is None:
            self._check_for_concurrent_obs(other)

        # get the DataFrames to combine
        new_df = other.get_data()
        old_df = self.get_data()

        # initialize an empty DataFrame containing all columns and indices
        new_columns = set(new_df.keys())
        old_columns = set(old_df.keys())
        columns = new_columns.union(old_columns)
        index = old_df.index.union(new_df.index)
        combined_df = pd.DataFrame(index=index, columns=columns)

        # combine the DataFrames
        for variable in columns:

            if variable in old_df.keys() and variable in new_df.keys():

                old_index = old_df[variable].index
                new_index = new_df[variable].index

                # fill the empty DataFrame with rows that are in the old DataFrame but not the new
                old_index_diff = old_index.difference(new_index)
                combined_df.loc[old_index_diff, variable] = old_df.loc[old_index_diff, variable]

                # fill the empty DataFrame with rows that are in the new DataFrame but not the old
                new_index_diff = new_index.difference(old_index)
                combined_df.loc[new_index_diff, variable] = new_df.loc[new_index_diff, variable]

                # handle the row intersection
                index_intersect = old_index.intersection(new_index)
                if keep_curr_obs:
                    combined_df.loc[index_intersect, variable] = old_df.loc[index_intersect, variable]
                else:
                    combined_df.loc[index_intersect, variable] = new_df.loc[index_intersect, variable]

            elif variable in old_df.keys():

                combined_df.ix[old_df.index, variable] = old_df[variable]

            elif variable in new_df.keys():

                combined_df.ix[new_df.index, variable] = new_df[variable]

        combined_df = combined_df.apply(pd.to_numeric, args=('ignore', ))

        data_origin = self._data_origin.copy(deep=True)
        combined_data_origin = data_origin.append(other._data_origin)
        combined_data_origin.drop_duplicates(inplace=True)
        combined_data_origin.reset_index(drop=True, inplace=True)

        return type(self)(combined_df, combined_data_origin)

    @staticmethod
    def create_data_origin(data_df, data_path):
        """Create an origin DataFrame

        :param data_df: DataFrame containing variables
        :param data_path: String or list containing origin information for the variables in the data_df
        :return: 
        """

        variables = list(data_df)

        if isinstance(data_path, str):
            data_path = [data_path]

        data = []
        for path in data_path:
            data.extend([[variable_name, path] for variable_name in variables])

        data_origin = pd.DataFrame(data=data, columns=['variable', 'origin'])
        return data_origin

    def drop_variables(self, variable_names):
        """

        :param variable_names: list-like parameter containing names of variables to drop
        :return:
        """

        # drop the columns containing the variables
        data = self._data.copy(deep=True)
        data.drop(variable_names, axis=1, errors='ignore', inplace=True)

        data_origin = self._data_origin.copy(deep=True)

        # drop the variable origin information
        for variable in variable_names:

            variable_row = data_origin['variable'] == variable
            data_origin = data_origin[~variable_row]

        return type(self)(data, data_origin)

    def equals(self, other):
        """

        :param other:
        :return:
        """

        data_self = self.get_data()
        data_other = other.get_data()

        data_origin_self = self.get_origin()
        data_origin_other = other.get_origin()

        return data_self.equals(data_other) and data_origin_self.equals(data_origin_other)

    def get_data(self, index_step=None, interpolate_index=None):
        """Returns a Pandas DataFrame containing managed data.

        If step is specified, the returned DataFrame is interpolated on the frequency between the first and
        last times in the managed data time range.

        If index is specified, the returned DataFrame is interpolated on the indices.

        If step and index are specified, index will be resampled with the frequency given by step.

        :param index_step:
        :param interpolate_index:
        :return:
        """

        # get a copy of the contained DataFrame
        variable_names = self.get_variable_names()
        df = self._data[variable_names]

        # resample_index or index is specified
        if interpolate_index is not None or index_step is not None:

            # if resample_index isn't specified, set it to the internal data index
            if interpolate_index is None:
                interpolate_index = self._data.index

            # if index_step is specified, get a new resample_index based on the step
            if index_step is not None:
                interpolate_index = np.array(interpolate_index)
                index_range = interpolate_index[-1] - interpolate_index[0]
                num_index = int(index_range / index_step) + 1
                interpolate_index = interpolate_index[0] + np.arange(num_index) * index_step
                interpolate_index = pd.Index(interpolate_index)
                if isinstance(interpolate_index, pd.DatetimeIndex):
                    interpolate_index.name = 'DateTime'

            # add a DataFrame with indices to interpolate, sort the values, drop duplicates if any, and get a
            # DataFrame on the resampled indices
            interpolate_df = df.append(pd.DataFrame(index=interpolate_index))
            interpolate_df.sort_index(kind='mergesort', inplace=True)
            resampled_df = interpolate_df.interpolate('index')
            resampled_df.drop_duplicates(keep='first', inplace=True)
            df = resampled_df.loc[interpolate_index]

        return df[variable_names]

    def get_origin(self):
        """Return a DataFrame containing the variable origins.

        :return:
        """

        return self._data_origin.copy(deep=True)

    def get_variable(self, variable_name):
        """Return the time series of the valid observations of the variable described by variable_name.

        Any NaN observations will not be returned.

        :param variable_name: Name of variable to return time series
        :type variable_name: str
        :return:
        """

        self._check_variable_name(variable_name)

        return pd.DataFrame(self._data.loc[:, variable_name])

    def get_variable_names(self):
        """Return a list of variable names.

        :return: List of variable names.
        """
        data_columns = list(self._data.columns)
        data_columns.sort()

        return list(self._data.keys())

    def get_variable_observation(self, variable_name, time, time_window_width=0, match_method='nearest'):
        """

        :param variable_name:
        :param time:
        :param time_window_width:
        :param match_method:
        :return:
        """

        self._check_variable_name(variable_name)

        if time_window_width == 0 and match_method == 'nearest':
            try:
                variable_observation = self._data.ix[time, variable_name]
            except KeyError as err:
                if err.args[0] == time:
                    variable_observation = None
                else:
                    raise err

        else:

            variable = self.get_variable(variable_name).dropna()

            # get the subset of times with the variable
            time_diff = timedelta(minutes=time_window_width / 2.)

            # match the nearest-in-time observation
            if match_method == 'nearest':
                try:
                    nearest_index = variable.index.get_loc(time, method='nearest', tolerance=time_diff)
                    nearest_observation = variable.ix[nearest_index]
                    variable_observation = nearest_observation.as_matrix()[0]
                except KeyError:
                    variable_observation = np.nan

            # get the mean observation
            elif match_method == 'mean':
                beginning_time = time - time_diff
                ending_time = time + time_diff
                time_window = (beginning_time < variable.index) & (variable.index <= ending_time)
                variable_near_time = variable.ix[time_window]
                variable_observation = variable_near_time.mean()

            else:
                msg = 'Unrecognized keyword value for match_method: {}'.format(match_method)
                raise ValueError(msg)

        return variable_observation

    def get_variable_origin(self, variable_name):
        """Get a list of the origin(s) for the given variable name.

        :param variable_name: Name of variable
        :type variable_name: str
        :return: List containing the origin(s) of the given variable.
        :return type: list
        """

        self._check_variable_name(variable_name)

        grouped = self._data_origin.groupby('variable')
        variable_group = grouped.get_group(variable_name)
        variable_origin = list(variable_group['origin'])

        return variable_origin

    def match_data(self, other, variable_name=None, time_window_width=0, match_method='nearest'):
        """

        :param other:
        :param variable_name:
        :param time_window_width:
        :param match_method:
        :return:
        """

        # initialize data for a DataManager
        matched_data = pd.DataFrame(index=self._data.index)
        variable_origin_data = []

        if variable_name is None:
            variable_names = other.get_variable_names()
        else:
            variable_names = [variable_name]

        for variable in variable_names:

            # skip adding the variable if it's in the constituent data set
            if variable in self.get_variable_names():
                continue

            # iterate through all rows and add the matched surrogate observation
            variable_series = pd.Series(index=matched_data.index, name=variable)
            for index, _ in matched_data.iterrows():
                observation_value = other.get_variable_observation(variable, index,
                                                                   time_window_width=time_window_width,
                                                                   match_method=match_method)
                variable_series[index] = observation_value

            # add the origins of the variable to the origin data list
            for origin in other.get_variable_origin(variable):
                variable_origin_data.append([variable, origin])

            # add the matched variable series to the dataframe
            matched_data[variable] = variable_series

        # create a data manager
        surrogate_variable_origin = pd.DataFrame(data=variable_origin_data, columns=['variable', 'origin'])
        matched_surrogate_data_manager = DataManager(matched_data, surrogate_variable_origin)

        # add the matched surrogate data manager to the constituent data manager
        return self.add_data_manager(matched_surrogate_data_manager)

    @classmethod
    def read_hdf(cls, path_or_buf, key):
        """

        :param path_or_buf:
        :param key:
        :return:
        """

        attribute_types = {'_data': pd,
                           '_data_origin': pd}

        if isinstance(path_or_buf, str):
            with pd.HDFStore(path_or_buf) as store:
                attributes = HDFio.read_hdf(store, attribute_types, key)
        else:
            attributes = HDFio.read_hdf(path_or_buf, attribute_types, key)

        result = cls.__new__(cls)

        for name, value in attributes.items():
            setattr(result, name, value)

        return result

    @classmethod
    def read_tab_delimited_data(cls, file_path):
        """Read a tab-delimited file containing a time series and return a DataManager instance.

        :param file_path: File path containing the TAB-delimited ASCII data file
        :param params: None.
        :return: DataManager object containing the data information
        """

        tab_delimited_df = cls._load_tab_delimited_data(file_path)

        origin = []

        for variable in tab_delimited_df.keys():
            origin.append([variable, file_path])

        data_origin = pd.DataFrame(data=origin, columns=['variable', 'origin'])

        return cls(tab_delimited_df, data_origin)

    def to_hdf(self, path_or_buf, key):
        """Write instance to an HDF file.

        :param path_or_buf: The path to an HDF file or an open HDFStore instance
        :param key: Identifier for the group in the HDF file
        :return:
        """

        attributes_dict = self.__dict__
        if isinstance(path_or_buf, str):
            with pd.HDFStore(path_or_buf) as store:
                HDFio.to_hdf(store, attributes_dict, key)
        else:
            HDFio.to_hdf(path_or_buf, attributes_dict, key)
