import numpy as np
import pandas as pd

from linearmodel.datamanager import DataManager
from linearmodel.model import find_raw_variable, get_exog_df, INVERSE_TRANSFORM_FUNCTIONS


def create_random_dataframe(number_of_rows=50):
    variable_names = ['a', 'b', 'c', 'd', 'e']
    data_size = (number_of_rows, 5)

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


def create_linear_model_test_data_set(response_variable, explanatory_variables, number_of_obs=50):
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
    test_data_origin = DataManager.create_data_origin(test_data_df, __file__)

    # return a DataManager with the regression data
    return DataManager(test_data_df, test_data_origin)
