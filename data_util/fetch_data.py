import numpy as np
from fairlearn.datasets import fetch_adult
from sklearn.model_selection import train_test_split


def split(data, target, sensitive, ratio=0.7, seed=876, sens_name='sex'):
    """
    Splits the data into training data and testing data
    :param data: input data
    :param target: target data
    :param sensitive: sensitive data
    :param ratio: ratio of data split (eg 0.7 is 70% training, 30% testing)
    :param seed: set pseudo-random seed so experiments can be repeated with same test/train split
    :param sens_name: name of sensitive label
    :return: x_tr is the training input, y_tr is the testing output, sens_tr is the training sensitive data input,
    x_te is the testing input, y_te is the testing output, sens_te is the testing sensitive data input
    """

    sens = sensitive[sens_name]
    x_tr, x_te, y_tr, y_te, sens_tr, sens_te = train_test_split(data, target, sens, train_size=ratio, random_state=seed)
    return x_tr, y_tr, sens_tr, x_te, y_te, sens_te


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

    return data.values, target.values, {sens: sensitive.values}, cat, bounds


def fetch_adult_data(sens='sex'):
    """
    Gets the adult income data set from the fairlearn package and returns it as standardised data

    :return: the feature values, the target values, dictionary for the sensitive values, list categorising each
    feature (continuous [c] or discrete [d]), pairs containing the lower and upper bound of each feature
    """

    # get the data set as a data frame
    (data, target) = fetch_adult(return_X_y=True, as_frame=True)

    # process the data in preperation to measure the robustness
    data, target, sensitive, cat, bounds = prepare_data(data, target, sens)

    return data, target, sensitive, cat, bounds


def add_noise(data, cat, bounds, iter=10, level=1):
    """
    Adds noise to the input data. Continuous data has laplacian noise added with mean 0 and var=level. Discrete data
    has level/len(data) of the values uniformly randomly selected from all possible values the variable could take
    :param data: Input data
    :param cat: category of data - [d] for discrete, [c] for continuous
    :param bounds: the upper and lower bound of each feature
    :param iter: Number of noisy instances we want with the selected level
    :param level: noise level
    :return: list of noisy data
    """

    # get indices of where continuous [c] data appears and discrete [d] data appears
    con_index = [None] * cat.count('c')
    dis_index = [None] * cat.count('d')

    # Find indices of discrete and continuous data
    con_index_count = 0
    dis_index_count = 0
    for index in range(0, len(cat)):
        if cat[index] == 'c':
            con_index[con_index_count] = index
            con_index_count += 1
        else:
            dis_index[dis_index_count] = index
            dis_index_count += 1

    # preallocate memory space to store the noisy data
    x_noise = [None] * iter

    for n in range(0, iter):
        # make copies of the data inputs
        x = data.copy()

        # use laplacian noise with mean 0 and specified noise level to add noise to continuous data
        x[:, con_index] += np.random.laplace(loc=0, scale=level, size=x[:, con_index].shape)

        # use bernoulli distribution to create noisy discrete data - 'level' represents the percentage of values for
        # a particular feature that will be randomly selected from a uniform distribution of all possible values
        for index in dis_index:
            num_of_changes = int(np.ceil(0.01 * level * len(x)))
            num_of_instances = np.arange(len(x))
            change_index = np.random.choice(num_of_instances, size=num_of_changes, replace=False)
            possible_values = data[:, index]
            x[change_index, index] = np.random.choice(possible_values, size=num_of_changes)

        x_noise[n] = x

    return x_noise


if __name__ == '__main__':
    data, target, sens, cat, bounds = fetch_adult_data()
    data_noise = add_noise(data, cat, bounds, iter=10, level=0.01)

    print('test done')
