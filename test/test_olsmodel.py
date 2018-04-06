import os
import sys
import unittest

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager
from linearmodel.model import INVERSE_TRANSFORM_FUNCTIONS, find_raw_variable, get_exog_df, MultipleOLSModel

current_path = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(current_path, 'data', 'model')


def create_test_data_set(response_variable, explanatory_variables, number_of_obs=50):
    """

    :param response_variable:
    :param explanatory_variables:
    :param number_of_obs:
    :return:
    """

    # find the raw explanatory variables and create a random DataFrame with the number of raw explanatory variables
    raw_explanatory_variables = list(set([raw_var for _, raw_var in
                                          [find_raw_variable(var) for var in explanatory_variables]]))
    explanatory_data = np.random.uniform(0.01, 10, size=(number_of_obs, len(raw_explanatory_variables)))
    explanatory_df = pd.DataFrame(data=explanatory_data, columns=raw_explanatory_variables)

    # get an exogenous DataFrame to calculate the response variable
    exog_df = get_exog_df(explanatory_df, explanatory_variables)

    # get the beta vector and error term
    number_of_parameters = len(explanatory_variables) + 1
    beta_vector = np.random.uniform(1, 10, size=(number_of_parameters, 1))
    error_term = np.random.normal(0, 0.1, size=(number_of_obs, 1))

    # calculate the response variable and create a DataFrame
    response_transform, raw_response_variable = find_raw_variable(response_variable)
    response_inverse_transform = INVERSE_TRANSFORM_FUNCTIONS[response_transform]
    response_data = response_inverse_transform(np.dot(exog_df, beta_vector) + error_term)
    response_df = pd.DataFrame(data=response_data, columns=[raw_response_variable])

    # create a DataFrame containing response and explanatory data
    test_data_df = pd.concat([response_df, explanatory_df], axis=1)

    # return a DataManager with the regression data
    return DataManager(test_data_df)


class TestMultipleOLSModelInit(unittest.TestCase):
    """Test the initialization of the MultipleOLSModel class"""

    def _save_test_case_data(self, model):

        test_case_name = sys._getframe(1).f_code.co_name

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
        response_variable = test_case_parameters['response_variable']
        explanatory_variables = test_case_parameters['explanatory_variables']
        model = MultipleOLSModel(data_set, response_variable=response_variable,
                                 explanatory_variables=explanatory_variables)

        # test the model form
        self.assertEqual(model.get_model_formula(), test_case_parameters['model_form'])

        # test that the model results are close enough to the expected results
        model_dataset = model.get_model_dataset()
        fitted_results = model_dataset[expected_fitted_results.keys()]
        is_close = np.isclose(fitted_results.as_matrix(), expected_fitted_results.as_matrix(), equal_nan=True)
        self.assertTrue(np.all(is_close))

    def test_model_init(self):
        """Test the successful initialization of a MultipleOLSModel instance"""

        test_case_parameters = {'response_variable': None,
                                'explanatory_variables': None,
                                'model_form': 'y ~ x1 + x2'}

        self._test_model_init(test_case_parameters)

    def test_model_init_specify_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when explanatory variables are specified and a
        response variable isn't
        """

        test_case_parameters = {'response_variable': None,
                                'explanatory_variables': ['y', 'x1'],
                                'model_form': 'x2 ~ y + x1'}

        self._test_model_init(test_case_parameters)

    def test_model_init_specify_response(self):
        """Test the initialization of a MultipleOLSModel instance when a response variable is specified and
        explanatory variables aren't
        """

        test_case_parameters = {'response_variable': 'x1',
                                'explanatory_variables': None,
                                'model_form': 'x1 ~ y + x2'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_response_and_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when explanatory and response variables are specified
        """

        test_case_parameters = {'response_variable': 'y',
                                'explanatory_variables': ['x1', 'x3'],
                                'model_form': 'y ~ x1 + x3'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_transformed_response(self):
        """Test the initialization of a MultipleOLSModel instance when specifying a transformed response variable."""

        test_case_parameters = {'response_variable': 'log10(y)',
                                'explanatory_variables': None,
                                'model_form': 'log10(y) ~ x1 + x2'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_one_transformed_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying one transformed explanatory
        variable.
        """

        test_case_parameters = {'response_variable': None,
                                'explanatory_variables': ['log10(x1)', 'x2'],
                                'model_form': 'y ~ log10(x1) + x2'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_all_transformed_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying two (all) transformed explanatory
        variables.
        """

        test_case_parameters = {'response_variable': None,
                                'explanatory_variables': ['log10(x1)', 'log10(x2)'],
                                'model_form': 'y ~ log10(x1) + log10(x2)'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_transformed_response_and_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying transformed response and explanatory
        variables.
        """

        test_case_parameters = {'response_variable': 'log10(y)',
                                'explanatory_variables': ['log10(x1)', 'log10(x2)'],
                                'model_form': 'log10(y) ~ log10(x1) + log10(x2)'}
        self._test_model_init(test_case_parameters)

    def test_model_init_multiple_explanatory_transform(self):
        """Test the initialization of a MultipleOLSModel instance when specifying multiple transformations of an
        explanatory variable with other explanatory variables.
        """

        test_case_parameters = {'response_variable': 'y',
                                'explanatory_variables': ['x1', 'log10(x1)', 'x2'],
                                'model_form': 'y ~ x1 + log10(x1) + x2'}
        self._test_model_init(test_case_parameters)

    def test_model_init_multiple_explanatory_transform_single(self):
        """Test the initialization of a MultipleOLSModel instance when specifying multiple transformations of a single
        explanatory variable.
        """

        test_case_parameters = {'response_variable': 'y',
                                'explanatory_variables': ['x1', 'log10(x1)'],
                                'model_form': 'y ~ x1 + log10(x1)'}
        self._test_model_init(test_case_parameters)


if __name__ == '__main__':
    unittest.main()
