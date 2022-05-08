import numpy as np
from fairlearn.datasets import fetch_adult


def get_data_type(data):
    """
    Takes in a data frame with headers and returns the data type (continuous [d] or discrete [c]) of each feature
    :param data: pandas data frame
    :return: Three lists. First list contains discrete labels, second list contains continuous labels, third list
    contains the order of label types eg ['c', 'd', 'd', ...]
    """

    # get number of labels and allocate memory to store list of label types
    labels = data.columns
    num_labels = len(np.asarray(labels))
    cat = [str] * num_labels

    # split labels into continuous list and discrete list
    cont_labels = np.asarray(data.select_dtypes('number').columns)
    dis_labels = np.asarray(data.select_dtypes('category').columns)

    # create list that defines all columns as either continuous 'c' or discrete 'd'
    for i in range(0, num_labels):
        if labels[i] in cont_labels:
            cat[i] = 'c'
        else:
            cat[i] = 'd'

    return dis_labels, cont_labels, cat


def standardise_data(data, dis_labels, con_labels):
    """
    standardises data such that discrete data is numeric and continuous data has mean 0 and variance 1
    :param data: original dataframe
    :param dis_labels: list of discrete labels
    :param con_labels: list of continuous labels
    :return: standardises data set
    """

    # standardise discrete labels such that the feature is numeric
    for label in dis_labels:
        data[label] = data[label].astype('category').cat.codes

    # standardise the continuous labels such that each feature has mean 0 and variance 1
    for label in con_labels:
        values = data[label]
        mean = values.mean()
        std = values.std()
        data[label] = ((values - mean) / std).tolist()

    return data


def get_bounds(data):
    """
    Gets the min and max values of each feature
    :param data: data frame
    :return: list of pairs of boundaries
    """

    # preallocate memory to hold boundaries
    labels = data.columns
    num_labels = len(labels)
    data_type = [float, float]
    bounds = [data_type] * num_labels

    # find boundaries for each label
    for i in range(0, num_labels):
        bounds[i] = [data[labels[i]].min(), data[labels[i]].max()]

    return bounds


def prepare_data(data, target, sens):
    """
    Prepares the data to have the robustness measured
    :param data: Input data frame
    :param target: Output data frame
    :param sens: Sensitive feature label
    :return: the original input data standardised with the sensitive feature removed, the original target data as
    numeric values, the sensitive data, the upper and lower bounds of each feature.
    """

    # split data into discrete features and continuous features
    dis_labels, cont_labels, cat = get_data_type(data)

    # standardise the data set (discrete data is numeric, continuous has mean 0 and variance 1)
    data = standardise_data(data, dis_labels, cont_labels)

    # save sensitive feature column and remove it from the data set, including from the categories list
    headers = list(data.columns)
    sens_index = headers.index(sens)
    del cat[sens_index]
    sensitive = data[sens]
    data = data.drop(sens, axis=1)

    # define the upper and lower bound of each feature column
    bounds = get_bounds(data)

    # standardise the target data
    target = target.astype('category').cat.codes

    return data, target, sensitive, cat, bounds


def fetch_adult_data():
    """
    Gets the adult income data set from the fairlearn package and returns it as standardised data

    :return: the feature values, the target values, dictionary for the sensitive values, list categorising each
    feature (continuous [c] or discrete [d]), pairs containing the lower and upper bound of each feature
    """

    # get the data set as a data frame
    (data, target) = fetch_adult(return_X_y=True, as_frame=True)

    # process the data in preperation to measure the robustness
    data, target, sensitive, cat, bounds = prepare_data(data, target, 'sex')

    return data.values, target.values, {'sex': sensitive.values}, cat, bounds


if __name__ == '__main__':
    input, target, sens, cat, bounds = fetch_adult_data()
