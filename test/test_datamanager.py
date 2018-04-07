import os
import unittest

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager

current_path = os.path.dirname(os.path.realpath(__file__))


def create_random_dataframe():
    variable_names = ['a', 'b', 'c', 'd', 'e']
    data_size = (50, 5)

    data = np.random.normal(size=data_size)
    data_df = pd.DataFrame(data=data, columns=variable_names)

    return data_df


def create_linspace_dataframe(data_start, data_stop, index_start, index_stop, columns, num_rows=6):

    # initialize a list containing the data for the first column
    data_list = [np.linspace(data_start, data_stop, num_rows)]

    # create data for the second to the last columns
    for i in range(1, len(columns)+1):
        column_data = i*data_list[i-1][-1] + (i+1)*data_list[i-1]
        data_list.append(column_data)

    # create a DataFrame with the column data
    data = dict(zip(columns, data_list))
    df = pd.DataFrame.from_dict(data)

    # add an index
    # if a TypeError is received, assume the index is np.datetime64
    try:
        index = np.linspace(index_start, index_stop, num_rows)
    except TypeError:
        index_range = index_stop - index_start
        datetime_step = index_range / (num_rows - 1)
        index = index_start + np.arange(num_rows) * datetime_step

    # set the index
    df.set_index(index, inplace=True)

    if isinstance(df.index, pd.DatetimeIndex):
        df.index.name = 'DateTime'

    return df


class TestDataManagerGetData(unittest.TestCase):
    """Test the get_data method of DataManager"""

    def test_get_data(self):
        """A simple test case of the DataManager.get_data method"""

        # create a DataManager from a DataFrame with random data
        data_df = create_random_dataframe()
        data_manager = DataManager(data_df)

        # get data from the DataManager
        results_df = data_manager.get_data()

        # test if the data manager returns a copy of the DataFrame
        pd.testing.assert_frame_equal(data_manager.get_data(), data_df)
        self.assertIsNot(results_df, data_df)

    def test_get_data_float_index_specify_step(self):
        """Test DataManager.get_data() a float type data index and the step parameter specified"""

        data_start = 0
        data_stop = 10
        index_start = 0.
        index_stop = 10.
        columns = ['x', 'y']
        num_rows = 6

        df1 = create_linspace_dataframe(data_start, data_stop, index_start, index_stop, columns, num_rows)
        dm = DataManager(df1)

        num_rows_1 = num_rows
        step_1 = (index_stop - index_start) / (num_rows_1 - 1)
        results_1 = dm.get_data(index_step=step_1)

        # with the same step, the DataFrames should be equal
        pd.testing.assert_frame_equal(results_1, df1)

        # get a linearly created DataFrame with twice the amount of rows, interpolate at the correct step, and
        # compare the results
        num_rows_2 = 2*num_rows_1 - 1
        df2 = create_linspace_dataframe(data_start, data_stop, index_start, index_stop, columns, num_rows*2 - 1)
        step_2 = (index_stop - index_start) / (num_rows_2 - 1)
        results_2 = dm.get_data(index_step=step_2)

        pd.testing.assert_frame_equal(results_2, df2)

    def test_get_data_datetime_index_specify_step(self):
        """Test DataManager.get_data() with a DateTimeIndex type data index and step parameter specified"""

        data_start = 0.
        data_stop = 10.
        index_start = index_start = np.datetime64('2018-01-01')
        index_step = np.timedelta64(15*60*1000, 'ms')
        num_rows = 7
        index_stop = index_start + index_step * num_rows
        columns = ['x', 'y']

        df1 = create_linspace_dataframe(data_start, data_stop, index_start, index_stop, columns, num_rows)
        dm = DataManager(df1)
        pd.testing.assert_frame_equal(dm.get_data(), df1)

        num_rows_2 = 12
        df2 = create_linspace_dataframe(data_start, data_stop, index_start, index_stop, columns, num_rows_2)
        index_step = df2.index[1] - df2.index[0]
        pd.testing.assert_frame_equal(dm.get_data(index_step=index_step), df2)


class TestDataManagerInit(unittest.TestCase):

    def test_init_with_origin(self):
        """Test the initialization method of DataManager with an origin DataFrame"""
        data_df = create_random_dataframe()
        variable_names = data_df.keys()

        origin_data = [[var, 'test'] for var in variable_names]
        origin_df = pd.DataFrame(data=origin_data, columns=['variable', 'origin'])
        data_manager_with_origin = DataManager(data_df, origin_df)

        # test that data is being stored correctly
        pd.testing.assert_frame_equal(data_manager_with_origin.get_data(), data_df)
        pd.testing.assert_frame_equal(data_manager_with_origin.get_origin(), origin_df)

        # test that DataFrames aren't the same instance
        self.assertFalse(data_manager_with_origin.get_data() is data_df)
        self.assertFalse(data_manager_with_origin.get_origin() is origin_df)

    def test_init_without_origin(self):
        """Test the initialization method of DataManager with no origin DataFrame"""
        data_df = create_random_dataframe()
        variable_names = data_df.keys()

        origin_data = [[var, np.nan] for var in variable_names]
        nan_origin_df = pd.DataFrame(data=origin_data, columns=['variable', 'origin'])
        data_manager_without_origin = DataManager(data_df)

        # test that data is being stored correctly
        pd.testing.assert_frame_equal(data_manager_without_origin.get_data(), data_df)
        pd.testing.assert_frame_equal(data_manager_without_origin.get_origin(), nan_origin_df)

        # test that DataFrames aren't the same instance
        self.assertFalse(data_manager_without_origin.get_data() is data_df)
        self.assertFalse(data_manager_without_origin.get_origin() is nan_origin_df)

    def test_init_read_tab_delimited_file_no_datetime(self):
        """Test the initialization of a DataManager instance using read_tab_delimited_data() from a file with no
        date/time information
        """

        # pick a data file from the model test data set
        test_data_file_path = os.path.join(current_path, 'data', 'model',
                                           'TestMultipleOLSModelInit', 'test_model_init.txt')

        # read the file into a DataManager
        data_manager = DataManager.read_tab_delimited_data(test_data_file_path)

        # read the file into a DataFrame
        df = pd.read_table(test_data_file_path, sep='\t')

        # make sure the DataManager's data and DataFrame are equal
        pd.testing.assert_frame_equal(data_manager.get_data(), df)


if __name__ == '__main__':
    unittest.main()
