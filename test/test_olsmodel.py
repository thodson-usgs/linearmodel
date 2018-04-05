import os
import sys
import unittest

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager
from linearmodel.model import MultipleOLSModel
from linearmodel.stats import ols_parameter_estimate

current_path = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(current_path, 'data', 'model')


def create_data_set(p=2, n=50, response_transform=None, explanatory_transform=None):
    """Create a data set with p coefficients and n observations

    :param p: Number of parameters (explanatory variables plus intercept)
    :param n: Number of observations
    :param response_transform: Transform of the response variable. None or 'log10'
    :param explanatory_transform: Transforms of the explanatory variables.
                                  None or dictionary containing index, transform items
    :return: DataManager containing model data
    """

    # create an explanatory variable matrix
    x = np.random.uniform(low=1, high=20, size=(n, p-1))

    # copy the explanatory variables for the exogenous matrix
    x_for_exog = x.copy()

    # transform the explanatory variables for the exogenous matrix
    if explanatory_transform is not None:
        for index, transform in explanatory_transform.items():
            if transform == 'log10':
                transform_increase_factor = 5
                x[:, index] = transform_increase_factor * x[:, index]
                x_for_exog[:, index] = np.log10(x[:, index])
            else:
                raise ValueError('Unrecognized explanatory transform')

    # add the intercept column to the explanatory variables matrix to get an exogenous matrix
    exog = np.append(np.ones((n, 1)), x_for_exog, axis=1)

    # create a response variable from the explanatory variables and parameters
    # apply the beta vector to the exogenous matrix and add the error term if the response is not transformed
    if response_transform is None:
        # create a parameter vector to get the response variable
        beta_high = 10
        beta_low = 1
        y_inverse = lambda var: var
    # if the response is transformed, get the transformed response, then calculate the raw
    elif response_transform == 'log10':
        beta_high = 5
        beta_low = 0.01
        y_inverse = lambda var: np.power(10, var)
    else:
        raise ValueError('Unrecognized response transform')

    # create a parameter vector, error terms, and y vector
    beta_vector = np.random.uniform(low=beta_low, high=beta_high, size=(p, 1))
    error_std = beta_vector[1:].max()/10
    error_term = np.random.normal(0, error_std, (n, 1))
    y = y_inverse(np.dot(exog, beta_vector) + error_term)

    # create a DataFrame from the response and explanatory data
    data = np.append(y, x, axis=1)
    columns = ['y'] + ['x{:1}'.format(i) for i in range(1, p)]
    df = pd.DataFrame(data=data, columns=columns)

    # return a DataManager
    return DataManager(df)


def estimate_parameters(dm, response_variable='y', explanatory_variables=None,
                        response_transform=None, explanatory_transform=None):
    """Estimate the parameters of a data set created by create_data_set

    :param dm: DataManager created by create_data_set
    :param response_variable:
    :param explanatory_variables: List of explanatory variables
    :param response_transform: Transform of the response variable. None or 'log10'
    :param explanatory_transform: Transforms of the explanatory variables.
                                  None or dictionary containing index, transform items
    :return:
    """
    df = dm.get_data()

    if response_transform is None:
        endog = df.as_matrix([response_variable])
    elif response_transform == 'log10':
        # transform the response variable
        endog = np.log10(df.as_matrix([response_variable]))

    # DataManager returns the DataFrame sorted by columns, so explanatory variables are first
    if explanatory_variables is None:
        exog_columns = df.columns != response_variable
        x = df.as_matrix(df.columns[exog_columns])
    else:
        exog_columns = explanatory_variables
        x = df[exog_columns].as_matrix()

    if explanatory_transform is not None:
        # transform the explanatory variables
        for index, transform in explanatory_transform.items():
            if transform == 'log10':
                x[:, index] = np.log10(x[:, index])

    n = x.shape[0]
    exog = np.append(np.ones((n, 1)), x, axis=1)

    parameter_estimate = ols_parameter_estimate(exog, endog)

    return parameter_estimate


class TestMultipleOLSModelInit(unittest.TestCase):
    """Test the initialization of the MultipleOLSModel class"""

    def _save_test_case_model_data(self, model_dataset):

        test_case_name = sys._getframe(1).f_code.co_name

        # find the test case data file path
        test_case_file_name = test_case_name + ".txt"
        test_data_path = os.path.join(model_path, self.__class__.__name__)
        test_case_file_path = os.path.join(test_data_path, test_case_file_name)

        model_dataset.to_csv(test_case_file_path, sep='\t')

    def _test_model_init(self, test_case_parameters):

        # print(sys._getframe(1).f_code.co_name)
        test_case_name = sys._getframe(1).f_code.co_name

        # find the test case data file path
        test_case_file_name = test_case_name + ".txt"
        test_data_path = os.path.join(model_path, self.__class__.__name__)
        test_case_file_path = os.path.join(test_data_path, test_case_file_name)

        # read the test case data and get the fitted values
        test_case_df = pd.read_table(test_case_file_path, dtype=np.float64)
        fitted_df = test_case_df.filter(regex='Fitted *')

        # drop the fitted values and create a data set to pass to the LinearModel
        data_set_df = test_case_df.drop(fitted_df.keys(), axis=1)
        data_set = DataManager(data_set_df)

        # initialize a model without specifying the response and explanatory variables
        response_variable = test_case_parameters['response_variable']
        explanatory_variables = test_case_parameters['explanatory_variables']
        model = MultipleOLSModel(data_set, response_variable=response_variable,
                                 explanatory_variables=explanatory_variables)

        # test the model form
        self.assertEqual(model.get_model_formula(), test_case_parameters['model_form'])

        model_dataset = model.get_model_dataset()
        fitted_results = model_dataset[fitted_df.keys()]
        pd.testing.assert_frame_equal(fitted_results, fitted_df)

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


if __name__ == '__main__':
    unittest.main()
