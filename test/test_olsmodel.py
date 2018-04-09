import os
import tempfile
import sys
import unittest

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager
from linearmodel.model import find_raw_variable, SimpleOLSModel, MultipleOLSModel
from test.test_util import create_linear_model_test_data_set

current_path = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(current_path, 'data', 'model')


class TestOLSModelInit(unittest.TestCase):
    """Base class for OLSModelInit tests"""

    def _create_test_case_data(self, test_case_parameters):
        """Create a dataset for assessing the test case defined by test_case_parameters"""
        response_variable = test_case_parameters['model_variables']['response_variable']
        try:
            explanatory_variables = test_case_parameters['model_variables']['explanatory_variables']
        except KeyError:
            explanatory_variables = [test_case_parameters['model_variables']['explanatory_variable']]
        data_set = create_linear_model_test_data_set(response_variable, explanatory_variables)
        model = test_case_parameters['test_class'](data_set, **test_case_parameters['init_kwargs'])
        self._save_test_case_data(model)

    def _save_test_case_data(self, model):
        """Only call from _create_test_case_data()"""

        test_case_name = sys._getframe(2).f_code.co_name

        # find the test case data file path
        test_case_file_name = test_case_name + ".txt"
        test_data_path = os.path.join(model_path, self.__class__.__name__)
        test_case_file_path = os.path.join(test_data_path, test_case_file_name)

        response_variable = model.get_response_variable()
        _, raw_response_variable = find_raw_variable(response_variable)

        explanatory_variables = model.get_explanatory_variables()
        raw_explanatory_variables = list(set([raw_var for _, raw_var in
                                              [find_raw_variable(var) for var in explanatory_variables]]))

        test_case_variables = [raw_response_variable] + raw_explanatory_variables + ['Fitted ' + response_variable]

        model_dataset = model.get_model_dataset()
        test_case_data = model_dataset[test_case_variables]

        try:
            test_case_data.to_csv(test_case_file_path, sep='\t', index=False)
        except FileNotFoundError:
            # if FileNotFoundError occurs, assume the directory doesn't exist, create it, and save again
            test_case_directory, _ = os.path.split(test_case_file_path)
            os.mkdir(test_case_directory)
            test_case_data.to_csv(test_case_file_path, sep='\t', index=False)

    def _test_model_init(self, test_case_parameters):

        # get the test case name
        # (the name of the calling method)
        test_case_name = sys._getframe(1).f_code.co_name

        # find the test case data file path
        test_case_file_name = test_case_name + ".txt"
        test_data_path = os.path.join(model_path, self.__class__.__name__)
        test_case_file_path = os.path.join(test_data_path, test_case_file_name)

        # read the test case data and get the fitted values
        test_case_df = pd.read_table(test_case_file_path, dtype=np.float64)
        expected_fitted_results = test_case_df.filter(regex='Fitted *')

        # drop the fitted values and create a data set to pass to the LinearModel
        data_set_df = test_case_df.drop(expected_fitted_results.keys(), axis=1)
        data_set = DataManager(data_set_df)

        # initialize a model without specifying the response and explanatory variables
        # response_variable = test_case_parameters['response_variable']
        # explanatory_variables = test_case_parameters['explanatory_variables']
        model = test_case_parameters['test_class'](data_set, **test_case_parameters['init_kwargs'])

        # test the model form
        self.assertEqual(model.get_model_formula(), test_case_parameters['model_form'])

        # test that the model results are close enough to the expected results
        model_dataset = model.get_model_dataset()
        fitted_results = model_dataset[expected_fitted_results.keys()]
        is_close = np.isclose(fitted_results.as_matrix(), expected_fitted_results.as_matrix(), equal_nan=True)
        self.assertTrue(np.all(is_close))


class TestMultipleOLSModelInit(TestOLSModelInit):
    """Test the initialization of instances of MultipleOLSModel class"""

    def test_model_init(self):
        """Test the successful initialization of a MultipleOLSModel instance"""

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': None, 'explanatory_variables': None},
                                'model_form': 'y ~ x1 + x2'}

        self._test_model_init(test_case_parameters)

    def test_model_init_specify_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when explanatory variables are specified and a
        response variable isn't
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': None, 'explanatory_variables': ['y', 'x1']},
                                'model_form': 'x2 ~ y + x1'}

        self._test_model_init(test_case_parameters)

    def test_model_init_specify_response(self):
        """Test the initialization of a MultipleOLSModel instance when a response variable is specified and
        explanatory variables aren't
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'x1', 'explanatory_variables': None},
                                'model_form': 'x1 ~ y + x2'}

        self._test_model_init(test_case_parameters)

    def test_model_init_specify_response_and_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when explanatory and response variables are specified
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'y', 'explanatory_variables': ['x1', 'x3']},
                                'model_form': 'y ~ x1 + x3'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_transformed_response(self):
        """Test the initialization of a MultipleOLSModel instance when specifying a transformed response variable."""

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'log10(y)', 'explanatory_variables': None},
                                'model_form': 'log10(y) ~ x1 + x2'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_one_transformed_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying one transformed explanatory
        variable.
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': ['log10(x1)', 'x2']},
                                'model_form': 'y ~ log10(x1) + x2'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_all_transformed_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying two (all) transformed explanatory
        variables.
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': ['log10(x1)', 'log10(x2)']},
                                'model_form': 'y ~ log10(x1) + log10(x2)'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_transformed_response_and_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying transformed response and explanatory
        variables.
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'log10(y)',
                                                'explanatory_variables': ['log10(x1)', 'log10(x2)']},
                                'model_form': 'log10(y) ~ log10(x1) + log10(x2)'}
        self._test_model_init(test_case_parameters)

    def test_model_init_multiple_explanatory_transform(self):
        """Test the initialization of a MultipleOLSModel instance when specifying multiple transformations of an
        explanatory variable with other explanatory variables.
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'y',
                                                'explanatory_variables': ['x1', 'log10(x1)', 'x2']},
                                'model_form': 'y ~ x1 + log10(x1) + x2'}
        self._test_model_init(test_case_parameters)

    def test_model_init_multiple_explanatory_transform_single(self):
        """Test the initialization of a MultipleOLSModel instance when specifying multiple transformations of a single
        explanatory variable.
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'y', 'explanatory_variables': ['x1', 'log10(x1)']},
                                'model_form': 'y ~ x1 + log10(x1)'}
        self._test_model_init(test_case_parameters)


class TestSimpleOLSModelInit(TestOLSModelInit):
    """Test the initialization of instances of SimpleOLSModel class"""

    def test_model_init(self):
        """Test the successful initialization of a SimpleOLSModel instance"""

        test_case_parameters = {'test_class': SimpleOLSModel,
                                'init_kwargs': {'response_variable': None, 'explanatory_variable': None},
                                'model_variables': {'response_variable': 'y', 'explanatory_variable': 'x'},
                                'model_form': 'y ~ x'}
        self._create_test_case_data(test_case_parameters)


class TestOLSModelHDF(unittest.TestCase):
    """Test HDF read/write functionality of OLSModel subclasses."""

    def setUp(self):
        """

        :return:
        """

        fd, temp_hdf_path = tempfile.mkstemp(suffix='.h5')
        os.close(fd)
        self.temp_hdf_path = temp_hdf_path

    def test_simpleolsmodel_hdf(self):
        """Test the functionality of the SimpleOLSModel.to_hdf() and read_hdf() methods"""

        key = '/'
        data = create_linear_model_test_data_set('y', ['x'])
        model = SimpleOLSModel(data, response_variable='y', explanatory_variable='x')
        model.to_hdf(self.temp_hdf_path, key)
        model_from_hdf = SimpleOLSModel.read_hdf(self.temp_hdf_path)
        self.assertTrue(model.equals(model_from_hdf))

    def tearDown(self):
        """

        :return:
        """
        if os.path.isfile(self.temp_hdf_path):
            os.remove(self.temp_hdf_path)


if __name__ == '__main__':
    unittest.main()
