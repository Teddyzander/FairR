import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_adult

def get_data_type(data):
    """
    Takes in a data frame with headers and returns the data type (continuous [d] or discrete [c]) of each feature
    :param data: pandas data frame
    :return: Three lists. First list contains discrete labels, second list contains continuous labels, third list
    contains the order of label types
    """

    # get number of labels and allocate memory to store list of label types
    labels = data.columns
    num_labels = len(np.asarray(labels))
    cat = [None] * num_labels

    # split labels into continuous list and discrete list
    cont_label = np.asarray(data.select_dtypes('number').columns)
    dis_label = np.asarray(data.select_dtypes('category').columns)

    # create list that defines all columns as either continuous 'c' or discrete 'd'
    for i in range(0, num_labels):
        if labels[i] in cont_label:
            cat[i] = 'c'
        else:
            cat[i] = 'd'

    return dis_label, cont_label, cat


def fetch_adult_data():
    """
    Gets the adult income data set from the fairlearn package

    :return: the feature values, the target values, dictionary for the sensitive features, list categorising each
    feature (continuous [c] or discrete [d], pairs containing the lower and upper bound of each feature
    """

    # get the data set as a data frame
    (data, target) = fetch_adult(return_X_y=True, as_frame=True)

    # split data into discrete features and continuous features
    dis_label, cont_label, cat = get_data_type(data)


if __name__ == '__main__':
    fetch_adult_data()
