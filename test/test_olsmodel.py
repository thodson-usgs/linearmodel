import unittest

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager
from linearmodel.model import MultipleOLSModel
from linearmodel.stats import ols_parameter_estimate


def create_data_set(p=2, n=50):
    """Create a data set with p coefficients and n observations

    :param p:
    :param n:
    :return:
    """

    beta_vector = np.random.uniform(low=0, high=5, size=(p, 1))

    x = np.random.uniform(low=0, high=10, size=(n, p-1))
    exog = np.append(np.ones((n, 1)), x, axis=1)

    error_std = 1
    error_term = np.random.normal(0, error_std, (n, 1))

    y = np.dot(exog, beta_vector) + error_term

    data = np.append(y, x, axis=1)
    columns = ['y'] + ['x{:1}'.format(i) for i in range(1, p)]
    df = pd.DataFrame(data=data, columns=columns)

    return DataManager(df)


def estimate_parameters(dm):
    """Estimate the parameters of a data set created by create_data_set

    :param dm: DataManager created by create_data_set
    :return:
    """
    df = dm.get_data()

    endog = df.as_matrix(['y'])

    # DataManager returns the DataFrame sorted by columns, so explanatory variables are first
    x = df.as_matrix(df.columns[:-1])
    n = x.shape[0]
    exog = np.append(np.ones((n, 1)), x, axis=1)

    parameter_estimate = ols_parameter_estimate(exog, endog)

    return parameter_estimate


class TestMultipleOLSModelInit(unittest.TestCase):
    """Test the initialization of the MultipleOLSModel class"""

    def test_model_init_no_transform(self):
        """Test the successful initialization of a MultipleOLSModel instance"""

        data_set_1 = create_data_set(p=3)

        # initialize a model without specifying the response and explanatory variables
        model = MultipleOLSModel(data_set_1)

        # test the model form
        self.assertEqual(model.get_model_formula(), 'y ~ x1 + x2')

        # test the parameters estimated by the model
        model_params = model.get_model_params()
        expected_parameters = estimate_parameters(data_set_1)
        param_is_close = np.isclose(model_params, expected_parameters)
        self.assertTrue(np.all(param_is_close))


if __name__ == '__main__':
    unittest.main()
