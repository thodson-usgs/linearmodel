import unittest

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager
from linearmodel.model import MultipleOLSModel
from linearmodel.stats import ols_parameter_estimate


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

    def test_model_init(self):
        """Test the successful initialization of a MultipleOLSModel instance"""

        data_set = create_data_set(p=3)

        # initialize a model without specifying the response and explanatory variables
        model = MultipleOLSModel(data_set)

        # test the model form
        self.assertEqual(model.get_model_formula(), 'y ~ x1 + x2')

        # test the parameters estimated by the model
        model_params = model.get_model_params()
        expected_parameters = estimate_parameters(data_set)
        param_is_close = np.isclose(model_params, expected_parameters)
        self.assertTrue(np.all(param_is_close))

    def test_model_init_specify_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when explanatory variables are specified and a
        response variable isn't
        """

        data_set = create_data_set(p=3)

        explanatory_variables = ['y', 'x1']
        model = MultipleOLSModel(data_set, explanatory_variables=explanatory_variables)

        # test the model form
        self.assertEqual(model.get_model_formula(), 'x2 ~ y + x1')

        # test the parameters estimated by the model
        model_params = model.get_model_params()
        expected_parameters = estimate_parameters(data_set, response_variable='x2')
        param_is_close = np.isclose(model_params, expected_parameters)
        self.assertTrue(np.all(param_is_close))

    def test_model_init_specify_response(self):
        """Test the initialization of a MultipleOLSModel instance when a response variable is specified and
        explanatory variables aren't
        """

        data_set = create_data_set(p=3)

        response_variable = 'x1'
        model = MultipleOLSModel(data_set, response_variable=response_variable)

        # test the model form
        self.assertEqual(model.get_model_formula(), 'x1 ~ y + x2')

        # test the parameters estimated by the model
        model_params = model.get_model_params()
        expected_parameters = estimate_parameters(data_set, response_variable=response_variable)
        param_is_close = np.isclose(model_params, expected_parameters)
        self.assertTrue(np.all(param_is_close))

    def test_model_init_specify_response_and_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when explanatory and response variables are specified
        """

        # get a data set with 3 explanatory variables but only specify two
        data_set = create_data_set(p=4)

        response_variale = 'y'
        explanatory_variables = ['x1', 'x3']
        model = MultipleOLSModel(data_set, response_variable=response_variale,
                                 explanatory_variables=explanatory_variables)

        # test the model form
        self.assertEqual(model.get_model_formula(), 'y ~ x1 + x3')

        # test the parameters estimated by the model
        model_params = model.get_model_params()
        expected_parameters = estimate_parameters(data_set, response_variable=response_variale,
                                                  explanatory_variables=explanatory_variables)
        param_is_close = np.isclose(model_params, expected_parameters)
        self.assertTrue(np.all(param_is_close))

    def test_model_init_specify_transformed_response(self):
        """Test the initialization of a MultipleOLSModel instance when specifying a transformed response variable."""

        data_set = create_data_set(p=3, response_transform='log10')

        response_variable = 'log10(y)'
        model = MultipleOLSModel(data_set, response_variable=response_variable)

        # test the model form
        self.assertEqual(model.get_model_formula(), 'log10(y) ~ x1 + x2')

        # test the parameters estimated by the model
        model_params = model.get_model_params()
        expected_parameters = estimate_parameters(data_set, response_transform='log10')
        param_is_close = np.isclose(model_params, expected_parameters)
        self.assertTrue(np.all(param_is_close))

    def test_model_init_specify_one_transformed_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying one transformed explanatory
        variable.
        """

        explanatory_transform = {0: 'log10'}
        data_set = create_data_set(p=3, explanatory_transform=explanatory_transform)
        explanatory_variables = ['log10(x1)', 'x2']
        model = MultipleOLSModel(data_set, explanatory_variables=explanatory_variables)

        # test the model form
        self.assertEqual(model.get_model_formula(), 'y ~ log10(x1) + x2')

        # test the parameters estimated by the model
        model_params = model.get_model_params()
        expected_parameters = estimate_parameters(data_set, explanatory_transform=explanatory_transform)
        param_is_close = np.isclose(model_params, expected_parameters)
        self.assertTrue(np.all(param_is_close))

    def test_model_init_specify_all_transformed_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying two (all) transformed explanatory
        variables.
        """

        explanatory_transform = {0: 'log10', 1: 'log10'}
        data_set = create_data_set(p=3, explanatory_transform=explanatory_transform)
        explanatory_variables = ['log10(x1)', 'log10(x2)']
        model = MultipleOLSModel(data_set, explanatory_variables=explanatory_variables)

        # test the model form
        self.assertEqual(model.get_model_formula(), 'y ~ log10(x1) + log10(x2)')

        # test the parameters estimated by the model
        model_params = model.get_model_params()
        expected_parameters = estimate_parameters(data_set, explanatory_transform=explanatory_transform)
        param_is_close = np.isclose(model_params, expected_parameters)
        self.assertTrue(np.all(param_is_close))

    def test_model_init_specify_transformed_response_and_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying transformed response and explanatory
        variables.
        """

        response_transform = 'log10'
        explanatory_transform = {0: 'log10', 1: 'log10'}
        data_set = create_data_set(p=3, response_transform=response_transform,
                                   explanatory_transform=explanatory_transform)

        response_variable = 'log10(y)'
        explanatory_variables = ['log10(x1)', 'log10(x2)']
        model = MultipleOLSModel(data_set, response_variable=response_variable,
                                 explanatory_variables=explanatory_variables)

        # test the model form
        self.assertEqual(model.get_model_formula(), 'log10(y) ~ log10(x1) + log10(x2)')

        # test the parameters estimated by the model
        model_params = model.get_model_params()
        expected_parameters = estimate_parameters(data_set, response_transform=response_transform,
                                                  explanatory_transform=explanatory_transform)
        param_is_close = np.isclose(model_params, expected_parameters)
        self.assertTrue(np.all(param_is_close))


if __name__ == '__main__':
    unittest.main()
