import numpy as np
import pandas as pd
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
    cat = [None] * num_labels

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
    bounds = [None] * num_labels

    # find boundaries for each label
    for i in range(0, num_labels):
        bounds[i] = [data[labels[i]].min(), data[labels[i]].max()]

    return bounds


def fetch_adult_data():
    """
    Gets the adult income data set from the fairlearn package and returns it as standardised data

    :return: the feature values, the target values, dictionary for the sensitive features, list categorising each
    feature (continuous [c] or discrete [d]), pairs containing the lower and upper bound of each feature
    """

    # get the data set as a data frame
    (data, target) = fetch_adult(return_X_y=True, as_frame=True)

    # split data into discrete features and continuous features
    dis_labels, cont_labels, cat = get_data_type(data)

    # standardise the data set
    data = standardise_data(data, dis_labels, cont_labels)

    # save sensitive feature column and remove it from the data set
    sensitive = data['sex']
    data = data.drop('sex', axis=1)

    # define the upper and lower bound of each feature column
    bounds = get_bounds(data)

    # standardise the target data
    target = target.astype('category').cat.codes

    return data.values, target.values, {'sex': sensitive.values}, cat, bounds


if __name__ == '__main__':
    fetch_adult_data()
