import copy
import os
import tempfile
import sys
import unittest

import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager
from linearmodel.model import find_raw_variable, ComplexOLSModel, CompoundLinearModel, SimpleOLSModel, MultipleOLSModel
from test.test_util import create_linear_model_test_data_set

current_path = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(current_path, 'data', 'model')


# TODO: Make data set writing optional and remove all test data from repo
# WRITE_DATA_SET = FALSE


class TestModel(unittest.TestCase):
    """Base class for OLSModelInit tests"""

    @staticmethod
    def _create_test_case_data(test_case_parameters, number_of_obs=50):
        """Create a data set for assessing the test case defined by test_case_parameters"""
        response_variable = test_case_parameters['model_variables']['response_variable']
        try:
            explanatory_variables = test_case_parameters['model_variables']['explanatory_variables']
        except KeyError:
            explanatory_variables = [test_case_parameters['model_variables']['explanatory_variable']]
        data_set = create_linear_model_test_data_set(response_variable, explanatory_variables, number_of_obs)
        return data_set

    def _create_and_save_test_case_data(self, test_case_parameters, number_of_obs=50):
        """Create and save a dataset for assessing the test case defined by test_case_parameters"""
        data_set = self._create_test_case_data(test_case_parameters, number_of_obs)
        model = test_case_parameters['test_class'](data_set, **test_case_parameters['init_kwargs'])
        self._save_test_case_data(model)

    @staticmethod
    def _find_test_name():
        """

        :return:
        """
        depth_limit = 15
        for i in range(depth_limit):
            test_case_name = sys._getframe(i).f_code.co_name
            if test_case_name[:4] == 'test':
                return test_case_name

        return None

    def _init_test_model(self, test_case_parameters):
        """Initialize and return a test model"""

        # get the test case name
        test_case_df = self._load_test_case_data()
        expected_fitted_results = test_case_df.filter(regex='Fitted *')

        # drop the fitted values and create a data set to pass to the LinearModel
        data_set_df = test_case_df.drop(expected_fitted_results.keys(), axis=1)
        data_set = DataManager(data_set_df)

        # initialize a model without specifying the response and explanatory variables
        model = test_case_parameters['test_class'](data_set, **test_case_parameters['init_kwargs'])

        return model

    def _load_test_case_data(self):

        test_case_name = self._find_test_name()

        # find the test case data file path
        test_case_file_name = test_case_name + ".txt"
        test_data_path = os.path.join(model_path, self.__class__.__name__)
        test_case_file_path = os.path.join(test_data_path, test_case_file_name)

        # read the test case data and get the fitted values
        test_case_df = pd.read_table(test_case_file_path, dtype=np.float64)

        return test_case_df

    def _save_test_case_data(self, model):

        test_case_name = self._find_test_name()

        # find the test case data file path
        test_case_file_name = test_case_name + ".txt"
        test_data_path = os.path.join(model_path, self.__class__.__name__)
        test_case_file_path = os.path.join(test_data_path, test_case_file_name)

        response_variable = model.get_response_variable()
        _, raw_response_variable = find_raw_variable(response_variable)

        explanatory_variables = model.get_explanatory_variables()
        try:
            raw_explanatory_variables = list(set([raw_var for _, raw_var in
                                                  [find_raw_variable(var) for var in explanatory_variables]]))
        # if TypeError exception, assume the model is a CompoundLinearModel and use the first segment's raw variable
        except TypeError:
            raw_explanatory_variables = list(set([raw_var for _, raw_var in
                                                  [find_raw_variable(var) for var in explanatory_variables[0]]]))

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

    def _test_model_fitted(self, model):

        # get the test case name
        test_case_df = self._load_test_case_data()
        expected_fitted_results = test_case_df.filter(regex='Fitted *')

        # test that the model results are close enough to the expected results
        model_dataset = model.get_model_dataset()
        fitted_results = model_dataset[expected_fitted_results.keys()]
        is_close = np.isclose(fitted_results.as_matrix(), expected_fitted_results.as_matrix(), equal_nan=True)
        self.assertTrue(np.all(is_close))


class TestModelInit(TestModel):

    def _test_model_init(self, test_case_parameters):

        # self._create_and_save_test_case_data(test_case_parameters, number_of_obs=10)

        model = self._init_test_model(test_case_parameters)

        # test the model form
        self.assertEqual(test_case_parameters['model_form'], model.get_model_formula())

        # test that the model results are close enough to the expected results
        self._test_model_fitted(model)

    def _test_model_init_raises_regex(self, test_case_parameters):
        """Test the failure of a model test case and assert that it raises an error

        :param test_case_parameters:
        :return:
        """
        data_set = self._create_test_case_data(test_case_parameters)
        self.assertRaisesRegex(test_case_parameters['expected_exception'], test_case_parameters['regex'],
                               test_case_parameters['test_class'], *[data_set], **test_case_parameters['init_kwargs'])


class TestComplexOLSModel(TestModel):
    """General test case for ComplexOLSModel"""

    class_test_case_parameters = {'test_class': ComplexOLSModel}

    def test_equals(self):
        """Test ComplexOLSModel.equals() and ComplexOLSModel.__eq__()"""
        test_case_parameters = {'model_variables': {'response_variable': 'y',
                                                    'explanatory_variables': ['x', 'sqrt(x)']}}
        test_data = self._create_test_case_data(test_case_parameters)
        model1 = ComplexOLSModel(test_data, response_variable='y', explanatory_variables=['x', 'sqrt(x)'])
        model2 = ComplexOLSModel(test_data, response_variable='y', explanatory_variables=['x', 'sqrt(x)'])
        model3 = ComplexOLSModel(test_data, response_variable='x', explanatory_variables=['y', 'sqrt(y)'])

        self.assertTrue(model1.equals(model2))
        self.assertTrue(model1 == model2)
        self.assertFalse(model1.equals(model3))

    def test_get_model_report(self):
        """Test ComplexOLSModel.get_model_report()"""
        test_case_parameters = {'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['x', 'log10(x)']}}
        test_case_parameters.update(self.class_test_case_parameters)
        test_set = self._create_test_case_data(test_case_parameters)
        model = ComplexOLSModel(test_set)
        model.set_explanatory_variables(test_case_parameters['model_variables']['explanatory_variables'])

        # test to see if this just doesn't raise an error for now
        model.get_model_report()
        # self._save_test_case_data(model)


class TestComplexOLSModelInit(TestModelInit):
    """Test the initialization of instances of ComplexOLSModel class"""

    class_test_case_parameters = {'test_class': ComplexOLSModel}

    def test_model_init(self):
        """Test the successful initialization of a ComplexOLSModel instance"""

        test_case_parameters = {'init_kwargs': {'response_variable': None, 'explanatory_variables': ['x', 'log10(x)']},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['x', 'log10(x)']},
                                'model_form': 'w ~ x + log10(x)'}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_fail_two_raw_variables(self):
        """Test the initialization of a ComplexOLSModel class with more than one raw explanatory variable."""
        test_case_parameters = {'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': ['x1', 'log10(x2)']},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['x1', 'log10(x2)']},
                                'expected_exception': ValueError,
                                'regex': 'x2 is not a transformation of x1'}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init_raises_regex(test_case_parameters)


class TestCompoundLinearModel(TestModel):
    """General test case for CompoundLinearModel"""

    class_test_case_parameters = {'test_class': CompoundLinearModel}

    @staticmethod
    def create_compound_test_data_set(response_variable, explanatory_variables, explanatory_ranges):
        """Create a test data set for CompoundLinearModel"""

        number_of_obs = 10

        data_set = create_linear_model_test_data_set(response_variable, explanatory_variables[0],
                                                     number_of_obs, explanatory_ranges[0])

        for i in range(1, len(explanatory_variables)):
            tmp_data_set = create_linear_model_test_data_set(response_variable, explanatory_variables[i],
                                                             number_of_obs, explanatory_ranges[i])
            tmp_df = tmp_data_set.get_data()
            tmp_range_index_start = data_set.get_data().index[-1] + 1
            tmp_range_index_stop = tmp_range_index_start + number_of_obs
            tmp_df.set_index(pd.RangeIndex(tmp_range_index_start, tmp_range_index_stop), inplace=True)
            tmp_data_set = DataManager(tmp_df, tmp_data_set.get_origin())
            data_set = data_set.add_data_manager(tmp_data_set)

        return data_set

    def test_equals(self):
        """Test CompoundLinearModel.equals() and CompoundLinearModel.__eq__()"""

        test_data = self.create_compound_test_data_set('y', [['x', 'sqrt(x)'], ['x']], [(0, 10), (10, 20)])
        model1 = CompoundLinearModel(test_data,
                                     response_variable='y',
                                     explanatory_variables=[['x', 'sqrt(x)'], ['x']],
                                     break_points=[10])
        model2 = CompoundLinearModel(test_data,
                                     response_variable='y',
                                     explanatory_variables=[['x', 'sqrt(x)'], ['x']],
                                     break_points=[10])
        model3 = CompoundLinearModel(test_data,
                                     response_variable='y',
                                     explanatory_variables=[['x']])
        self.assertTrue(model1.equals(model2))
        self.assertTrue(model1 == model2)
        self.assertFalse(model1.equals(model3))

    def test_get_model_report(self):
        """Test CompoundLinearModel.get_model_report()"""

        test_case_parameters = {'init_kwargs': {}}
        test_case_parameters.update(self.class_test_case_parameters)
        # test_set = self._create_compound_test_data_set('w', [['x', 'log10(x)']], [(0, 20)])
        # model = CompoundLinearModel(test_set)
        model = self._init_test_model(test_case_parameters)
        model.set_explanatory_variables([['x', 'log10(x)']])

        # test to see if this just doesn't raise an error for now
        model.get_model_report()

        model.set_break_points(10)
        model.get_model_report()

        # self._save_test_case_data(model)

    def test_predict_response_variable_non_transformed_response(self):
        """Test CompoundLinearModel.predict_response_variable() with a non-transformed response variable"""
        test_set = self.create_compound_test_data_set('w', [['x', 'sqrt(x)'], ['x']], [(0, 10), (10, 20)])
        model = CompoundLinearModel(test_set)
        model.set_break_points([10])
        model.set_explanatory_variables([['x', 'sqrt(x)'], ['x']])

        predicted_response_no_bias_correction = model.predict_response_variable(bias_correction=False)
        predicted_response_bias_correction = model.predict_response_variable(bias_correction=True)

        pd.testing.assert_frame_equal(predicted_response_bias_correction,
                                      predicted_response_no_bias_correction)

    def test_set_break_points(self):
        """Test CompoundLinearModel.set_break_points()"""
        test_case_parameters = {'init_kwargs': {}}
        test_case_parameters.update(self.class_test_case_parameters)

        # initialize a test model and set the break points
        model = self._init_test_model(test_case_parameters)
        model.set_break_points([10])

        # test the results
        self.assertEqual(model.get_model_formula(), ['w ~ x', 'w ~ x'])  # test the model segment formulas
        self.assertTrue(np.all(model.get_break_points() == np.array([-np.inf, 10, np.inf])))  # test the break points
        self.assertEqual(model.get_explanatory_variables(), [['x'], ['x']])  # test the explanatory variables
        self._test_model_fitted(model)  # test the model fit

    def test_set_break_points_set_explanatory_variables(self):
        """Test the combination CompoundLinearModel.set_break_points() and
        CompoundLinearModel.set_explanatory_variables()
        """
        test_case_parameters = {'init_kwargs': {}}
        test_case_parameters.update(self.class_test_case_parameters)

        # test_data = self._create_compound_test_data_set('w', [['x', 'sqrt(x)'], ['x']], [(0, 10), (10, 20)])
        # model = CompoundLinearModel(test_data)
        model = self._init_test_model(test_case_parameters)
        model.set_break_points([10])
        model.set_explanatory_variables([['x', 'sqrt(x)'], ['x']])

        self.assertEqual(model.get_model_formula(), ['w ~ x + sqrt(x)', 'w ~ x'])  # test the model segment formulas
        self.assertTrue(np.all(model.get_break_points() == np.array([-np.inf, 10, np.inf])))  # test the break points
        self.assertEqual(model.get_explanatory_variables(), [['x', 'sqrt(x)'], ['x']])  # test the explanatory vars
        # self._save_test_case_data(model)
        self._test_model_fitted(model)  # test the model fit

    def test_set_explanatory_variables(self):
        """Test CompoundLinearModel.set_explanatory_variables() with no break points"""
        test_case_parameters = {'init_kwargs': {}}
        test_case_parameters.update(self.class_test_case_parameters)

        # test_data = self._create_compound_test_data_set('w', [['x', 'log10(x)']], [(0, 10)])
        # model = CompoundLinearModel(test_data)
        model = self._init_test_model(test_case_parameters)

        model.set_explanatory_variables([['x', 'log10(x)']])

        self.assertEqual(model.get_model_formula(), ['w ~ x + log10(x)'])  # test the model segment formulas
        self.assertTrue(np.all(model.get_break_points() == np.array([-np.inf, np.inf])))  # test the break points
        self.assertEqual(model.get_explanatory_variables(), [['x', 'log10(x)']])  # test the explanatory vars
        # self._save_test_case_data(model)
        self._test_model_fitted(model)  # test the model fit

    def test_set_multiple_break_points(self):
        """Test CompoundLinearModel.set_break_points(), passing multiple break points"""
        test_case_parameters = {'init_kwargs': {}}
        test_case_parameters.update(self.class_test_case_parameters)

        model = self._init_test_model(test_case_parameters)
        model.set_break_points([10, 20])

        self.assertEqual(model.get_model_formula(), ['w ~ x', 'w ~ x', 'w ~ x'])  # test the model segment formulas
        self.assertTrue(np.all(model.get_break_points() == np.array([-np.inf, 10, 20, np.inf])))  # test break points
        self.assertEqual(model.get_explanatory_variables(), [['x'], ['x'], ['x']])  # test the explanatory variables
        self._test_model_fitted(model)  # test the model fit


class TestCompoundLinearModelInit(TestModelInit):
    """Test the initialization of instances of CompoundLinear class"""

    class_test_case_parameters = {'test_class': CompoundLinearModel}

    def _save_test_case_data(self, model):

        test_case_name = self._find_test_name()

        # find the test case data file path
        test_case_file_name = test_case_name + ".txt"
        test_data_path = os.path.join(model_path, self.__class__.__name__)
        test_case_file_path = os.path.join(test_data_path, test_case_file_name)

        response_variable = model.get_response_variable()
        _, raw_response_variable = find_raw_variable(response_variable)

        explanatory_variables = [var for segment_vars in model.get_explanatory_variables() for var in segment_vars]
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

    def test_model_init(self):
        """Test the initialization of a CompoundLinearModel instance for a simple case"""
        test_case_parameters = {'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': None},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['x1']},
                                'model_form': ['w ~ x1']}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_explanatory(self):
        """Test the initialization of a CompoundLinearModel instance with specifying the explanatory variable"""
        test_case_parameters = {'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': [['x1']]},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['x1']},
                                'model_form': ['w ~ x1']}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_response(self):
        """Test the initialization of a CompoundLinearModel instance with specifying the response variable"""
        test_case_parameters = {'init_kwargs': {'response_variable': 'y',
                                                'explanatory_variables': None},
                                'model_variables': {'response_variable': 'y',
                                                    'explanatory_variables': ['x1']},
                                'model_form': ['y ~ x1']}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_multiple_explanatory(self):
        """Test the initialization of a CompoundLinearModel instance with specifying multiple explanatory variables"""
        test_case_parameters = {'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': [['x1', 'log10(x1)']]},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['x1', 'log10(x1)']},
                                'model_form': ['w ~ x1 + log10(x1)']}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_multiple_explanatory_with_response(self):
        """Test the initialization of a CompoundLinearModel instance with specifying a response variable and multiple
        explanatory variables"""
        test_case_parameters = {'init_kwargs': {'response_variable': 'y',
                                                'explanatory_variables': [['x1', 'log10(x1)']]},
                                'model_variables': {'response_variable': 'y',
                                                    'explanatory_variables': ['x1', 'log10(x1)']},
                                'model_form': ['y ~ x1 + log10(x1)']}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_break_point(self):
        """Test the initialization of a CompoundLinearModel instance with specifying a break point"""
        test_case_parameters = {'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': None,
                                                'break_points': [5]},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['x1']},
                                'model_form': ['w ~ x1', 'w ~ x1']}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_break_point_explanatory_variables(self):
        """Test the initialization of a CompoundLinearModel instance with specifying a break point and explanatory
        variables"""
        test_case_parameters = {'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': [['x1'], ['x1']],
                                                'break_points': [5]},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['x1']},
                                'model_form': ['w ~ x1', 'w ~ x1']}
        test_case_parameters.update(self.class_test_case_parameters)
        # self._create_and_save_test_case_data(test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_break_point_with_explanatory_and_response(self):
        """Test the initialization of a CompoundLinearModel instance with specifying a break point and explanatory and
        response variables"""
        test_case_parameters = {'init_kwargs': {'response_variable': 'y',
                                                'explanatory_variables': [['x1'], ['x1']],
                                                'break_points': [5]},
                                'model_variables': {'response_variable': 'y',
                                                    'explanatory_variables': ['x1']},
                                'model_form': ['y ~ x1', 'y ~ x1']}
        test_case_parameters.update(self.class_test_case_parameters)
        # self._create_and_save_test_case_data(test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_break_point_with_transform_explanatory_and_response(self):
        """Test the initialization of a CompoundLinearModel instance with specifying a break point and transformed
        explanatory variables and response variable"""
        test_case_parameters = {'init_kwargs': {'response_variable': 'y',
                                                'explanatory_variables': [['x1', 'log10(x1)'], ['x1', 'log10(x1)']],
                                                'break_points': [5]},
                                'model_variables': {'response_variable': 'y',
                                                    'explanatory_variables': ['x1', 'log10(x1)']},
                                'model_form': ['y ~ x1 + log10(x1)', 'y ~ x1 + log10(x1)']}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_fail_on_multiple_raw_explanatory_variables(self):
        """Test the failure of the initialization of a CompoundLinearModel instance with specifying multiple raw
        explanatory variables
        """

        test_case_parameters = {'init_kwargs': {'response_variable': 'y',
                                                'explanatory_variables': [['x1', 'log10(x2)'], ['x1', 'log10(x2)']],
                                                'break_points': [5]},
                                'model_variables': {'response_variable': 'y',
                                                    'explanatory_variables': ['x1', 'log10(x2)']},
                                'expected_exception': ValueError,
                                'regex': "x2 is not a transformation of x1."}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init_raises_regex(test_case_parameters)


class TestMultipleOLSModelInit(TestModelInit):
    """Test the initialization of instances of MultipleOLSModel class"""

    class_test_case_parameters = {'test_class': MultipleOLSModel}

    def test_model_init(self):
        """Test the successful initialization of a MultipleOLSModel instance"""

        test_case_parameters = {'init_kwargs': {'response_variable': None, 'explanatory_variables': None},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['x1', 'x2']},
                                'model_form': 'w ~ x1 + x2'}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_multiple_explanatory_transform(self):
        """Test the initialization of a MultipleOLSModel instance when specifying multiple transformations of an
        explanatory variable with other explanatory variables.
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'y',
                                                'explanatory_variables': ['x1', 'log10(x1)', 'x2']},
                                'model_variables': {'response_variable': 'y',
                                                    'explanatory_variables': ['x1', 'log10(x1)', 'x2']},
                                'model_form': 'y ~ x1 + log10(x1) + x2'}
        self._test_model_init(test_case_parameters)

    def test_model_init_multiple_explanatory_transform_single(self):
        """Test the initialization of a MultipleOLSModel instance when specifying multiple transformations of a single
        explanatory variable.
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'y',
                                                'explanatory_variables': ['x1', 'log10(x1)']},
                                'model_variables': {'response_variable': 'y',
                                                    'explanatory_variables': ['x1', 'log10(x1)']},
                                'model_form': 'y ~ x1 + log10(x1)'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_all_transformed_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying two (all) transformed explanatory
        variables.
        """

        test_case_parameters = {'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': ['log10(x1)', 'log10(x2)']},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['log10(x1)', 'log10(x2)']},
                                'model_form': 'w ~ log10(x1) + log10(x2)'}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when explanatory variables are specified and a
        response variable isn't
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': ['y', 'x1']},
                                'model_variables': {'response_variable': 'x2',
                                                    'explanatory_variables': ['y', 'x1']},
                                'model_form': 'x2 ~ y + x1'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_response(self):
        """Test the initialization of a MultipleOLSModel instance when a response variable is specified and
        explanatory variables aren't
        """

        test_case_parameters = {'init_kwargs': {'response_variable': 'x1', 'explanatory_variables': None},
                                'model_variables': {'response_variable': 'x1',
                                                    'explanatory_variables': ['y', 'x2']},
                                'model_form': 'x1 ~ x2 + y'}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_response_and_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when explanatory and response variables are specified
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'y',
                                                'explanatory_variables': ['x1', 'x3']},
                                'model_variables': {'response_variable': 'y',
                                                    'explanatory_variables': ['x1', 'x3']},
                                'model_form': 'y ~ x1 + x3'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_transformed_response(self):
        """Test the initialization of a MultipleOLSModel instance when specifying a transformed response variable."""

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'log10(y)',
                                                'explanatory_variables': None},
                                'model_variables': {'response_variable': 'log10(y)',
                                                    'explanatory_variables': ['x1', 'x2']},
                                'model_form': 'log10(y) ~ x1 + x2'}
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_one_transformed_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying one transformed explanatory
        variable.
        """

        test_case_parameters = {'init_kwargs': {'response_variable': None,
                                                'explanatory_variables': ['log10(x1)', 'x2']},
                                'model_variables': {'response_variable': 'w',
                                                    'explanatory_variables': ['log10(x1)', 'x2']},
                                'model_form': 'w ~ log10(x1) + x2'}
        test_case_parameters.update(self.class_test_case_parameters)
        self._test_model_init(test_case_parameters)

    def test_model_init_specify_transformed_response_and_explanatory(self):
        """Test the initialization of a MultipleOLSModel instance when specifying transformed response and explanatory
        variables.
        """

        test_case_parameters = {'test_class': MultipleOLSModel,
                                'init_kwargs': {'response_variable': 'log10(y)',
                                                'explanatory_variables': ['log10(x1)', 'log10(x2)']},
                                'model_variables': {'response_variable': 'log10(y)',
                                                    'explanatory_variables': ['log10(x1)', 'log10(x2)']},
                                'model_form': 'log10(y) ~ log10(x1) + log10(x2)'}
        self._test_model_init(test_case_parameters)


class TestSimpleOLSModel(TestModel):
    """General test case for SimpleOLSModel"""

    def test_equals(self):
        """Test SimpleOLSModel.equals() and SimpleOLSModel.__eq__()"""
        test_case_parameters = {'model_variables': {'response_variable': 'y', 'explanatory_variable': 'x'}}
        test_data = self._create_test_case_data(test_case_parameters)
        model1 = SimpleOLSModel(test_data, response_variable='y', explanatory_variable='x')
        model2 = SimpleOLSModel(test_data, response_variable='y', explanatory_variable='x')
        model3 = SimpleOLSModel(test_data, response_variable='x', explanatory_variable='y')

        self.assertTrue(model1.equals(model2))
        self.assertTrue(model1 == model2)
        self.assertFalse(model1 == model3)

    def test_predict_response_variable_with_transformed_response(self):
        """Test SimpleOLSModel.predict_response_variable() with a transformed response variable"""
        test_case_parameters = {'model_variables': {'response_variable': 'w', 'explanatory_variables': ['x']}}
        test_set = self._create_test_case_data(test_case_parameters)
        model = SimpleOLSModel(test_set, response_variable='log10(w)', explanatory_variable='log10(x)')

        model_dataset = model.get_model_dataset()
        fitted_response = model_dataset['Fitted log10(w)']

        # test without bias correction
        no_bias_correction_df = model.predict_response_variable(bias_correction=False, raw_response=True)
        is_close_no_bcf = np.isclose(10**fitted_response.values, no_bias_correction_df['w'].values)
        self.assertTrue(np.all(is_close_no_bcf))

        # test with bias correction
        bias_correction_df = model.predict_response_variable(bias_correction=True, raw_response=True)
        bias_correction_factor = (10**model_dataset['Raw Residual'].values).mean()
        is_close_bcf = np.isclose(bias_correction_factor*10**fitted_response.values, bias_correction_df['w'].values)
        self.assertTrue(np.all(is_close_bcf))


class TestSimpleOLSModelInit(TestModelInit):
    """Test the initialization of instances of SimpleOLSModel class"""

    def test_model_init(self):
        """Test the successful initialization of a SimpleOLSModel instance"""

        test_case_parameters = {'test_class': SimpleOLSModel,
                                'init_kwargs': {'response_variable': None, 'explanatory_variable': None},
                                'model_variables': {'response_variable': 'w', 'explanatory_variable': 'x'},
                                'model_form': 'w ~ x'}
        self._test_model_init(test_case_parameters)


class TestLinearModelHDF(unittest.TestCase):
    """Test HDF read/write functionality of OLSModel subclasses."""

    def setUp(self):
        """

        :return:
        """

        fd, temp_hdf_path = tempfile.mkstemp(suffix='.h5')
        os.close(fd)
        self.temp_hdf_path = temp_hdf_path

    def test_compoundlinearmodel_hdf(self):
        """Test the functionality of the CompoundLinearModel.to_hdf() and read_hdf() methods"""
        key = '/CompoundLinearModelHDFtest'
        data = TestCompoundLinearModel.create_compound_test_data_set('y', [['x', 'sqrt(x)'], ['x']],
                                                                     [(0, 10), (10, 20)])
        model = CompoundLinearModel(data, response_variable='y', explanatory_variables=[['x', 'sqrt(x)'], ['x']],
                                    break_points=[10])
        model.to_hdf(self.temp_hdf_path, key)
        model_from_hdf = CompoundLinearModel.read_hdf(self.temp_hdf_path, key)
        self.assertTrue(model.equals(model_from_hdf))

    def test_complexolsmodel_hdf(self):
        """Test the functionality of the ComplexOLSModel.to_hdf() and read_hdf() methods"""

        key = '/ComplexOLSModelHDFtest'
        data = create_linear_model_test_data_set('y', ['x', 'log10(x)'])
        model = ComplexOLSModel(data, response_variable='y', explanatory_variables=['x', 'log10(x)'])
        model.to_hdf(self.temp_hdf_path, key)
        model_from_hdf = ComplexOLSModel.read_hdf(self.temp_hdf_path, key)
        self.assertTrue(model.equals(model_from_hdf))

    def test_simpleolsmodel_hdf(self):
        """Test the functionality of the SimpleOLSModel.to_hdf() and read_hdf() methods"""

        key = '/SimpleOLSModelHDFtest'
        data = create_linear_model_test_data_set('y', ['x'])
        model = SimpleOLSModel(data, response_variable='y', explanatory_variable='x')
        model.to_hdf(self.temp_hdf_path, key)
        model_from_hdf = SimpleOLSModel.read_hdf(self.temp_hdf_path, key)
        self.assertTrue(model.equals(model_from_hdf))

    def test_multipleolsmodel_hdf(self):
        """Test the functionality of the SimpleOLSModel.to_hdf() and read_hdf() methods"""
        key = '/MultipleOLSModelHDFtest'
        data = create_linear_model_test_data_set('y', ['x1', 'x2'])
        model = MultipleOLSModel(data, response_variable='y', explanatory_variables=['x1', 'x2'])
        model.to_hdf(self.temp_hdf_path, key)
        model_from_hdf = MultipleOLSModel.read_hdf(self.temp_hdf_path, key)
        self.assertTrue(model.equals(model_from_hdf))

    def tearDown(self):
        """

        :return:
        """
        if os.path.isfile(self.temp_hdf_path):
            os.remove(self.temp_hdf_path)


if __name__ == '__main__':
    unittest.main()
