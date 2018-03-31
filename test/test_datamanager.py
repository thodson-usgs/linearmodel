import unittest

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager


class TestDataManager(unittest.TestCase):

    @staticmethod
    def get_data_frame_a():
        variable_names = ['a', 'b', 'c', 'd', 'e']
        data_size = (50, 5)

        data = np.random.normal(size=data_size)
        data_df = pd.DataFrame(data=data, columns=variable_names)
        data_df.index.name = 'DateTime'

        return data_df

    def test_init_with_origin(self):
        """Test the initialization method of DataManager with an origin DataFrame"""
        data_df = self.get_data_frame_a()
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
        data_df = self.get_data_frame_a()
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


if __name__ == '__main__':
    unittest.main()
