import copy
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager
from test.test_util import create_random_dataframe, create_linspace_dataframe

current_path = os.path.dirname(os.path.realpath(__file__))


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
        df2 = create_linspace_dataframe(data_start, data_stop, index_start, index_stop, columns, num_rows * 2 - 1)
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


class TestDataManager(unittest.TestCase):
    """Test miscellaneous features of the DataManager class"""

    def setUp(self):
        """

        :return:
        """

        fd, temp_hdf_path = tempfile.mkstemp(suffix='.h5')
        os.close(fd)
        self.temp_hdf_path = temp_hdf_path

    def test_add_data_manager_simple(self):
        """Test a simple case of DataManager.add_data_manager() usage"""

        # create a data manager with random data
        df = create_random_dataframe(number_of_rows=50)
        dm = DataManager(df)

        # split the data up and add the results together
        dm1 = DataManager(df.iloc[:25])
        dm2 = DataManager(df.iloc[25:])
        dm_add_result = dm1.add_data_manager(dm2)

        # test that the original data set and the add result are equal
        self.assertTrue(dm.equals(dm_add_result))

    def test_deepcopy(self):
        """Test the deepcopy functionality of instances of the DataManager class"""
        df = create_random_dataframe()
        dm1 = DataManager(df)
        dm2 = copy.deepcopy(dm1)
        self.assertTrue(dm1.equals(dm2))
        self.assertIsNot(dm1, dm2)

    def test_equals(self):
        """Test the DataManager.equals() and __eq__() methods"""
        df1 = create_random_dataframe()
        dm1 = DataManager(df1)
        dm2 = DataManager(df1)

        self.assertTrue(dm1.equals(dm2))
        self.assertTrue(dm1 == dm2)

        df3 = create_random_dataframe()
        dm3 = DataManager(df3)

        self.assertFalse(dm1.equals(dm3))
        self.assertFalse(dm1 == dm3)
        self.assertTrue(dm1 != dm3)

    def test_match_data(self):
        """

        :return:
        """

        test_data_path = os.path.join(current_path, 'data', 'datamanager', 'test_match_data')
        dm1_path = os.path.join(test_data_path, 'dm1.txt')
        dm1 = DataManager.read_tab_delimited_data(dm1_path)

        dm2_path = os.path.join(test_data_path, 'dm2.txt')
        dm2 = DataManager.read_tab_delimited_data(dm2_path)

        dm3 = dm1.match_data(dm2, time_window_width=30, match_method='mean')

        expected_data_path = os.path.join(test_data_path, 'dm3.txt')
        expected_data = pd.read_csv(expected_data_path, index_col='DateTime', sep='\t', parse_dates=True)

        self.assertTrue(np.allclose(dm3.get_data().as_matrix(), expected_data.as_matrix(), atol=1e-4))

    def test_to_hdf_buf(self):
        """Test the DataManager.to_hdf() and DataManager.read_hdf() methods when saving with a pd.HDFStore instance"""

        key = '/dm/'

        data = create_random_dataframe()
        data_origin = DataManager.create_data_origin(data, 'test')
        dm1 = DataManager(data, data_origin)

        with pd.HDFStore(self.temp_hdf_path) as store:
            dm1.to_hdf(store, key)

        with pd.HDFStore(self.temp_hdf_path) as store:
            dm2 = DataManager.read_hdf(store, key)

        self.assertTrue(dm1.equals(dm2))

    def test_to_hdf_path(self):
        """Test the DataManager.to_hdf() and DataManager.read_hdf() methods when saving with a file path"""

        key = '/dm'

        data = create_random_dataframe()
        data_origin = DataManager.create_data_origin(data, 'test')
        dm1 = DataManager(data, data_origin)
        dm1.to_hdf(self.temp_hdf_path, key)

        dm2 = DataManager.read_hdf(self.temp_hdf_path, key)

        self.assertTrue(dm1.equals(dm2))

    def tearDown(self):
        """

        :return:
        """

        if os.path.isfile(self.temp_hdf_path):
            os.remove(self.temp_hdf_path)


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
        df.sort_index(axis=1, inplace=True)

        # make sure the DataManager's data and DataFrame are equal
        pd.testing.assert_frame_equal(data_manager.get_data(), df)


if __name__ == '__main__':
    unittest.main()
