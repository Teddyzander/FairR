import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_adult

def fetch_adult_data():
    """
    Gets the adult income data set from the fairlearn package

    :return: the feature values, the target values, dictionary for the sensitive features, list categorising each
    feature (continuous [c] or discrete [d], pairs containing the lower and upper bound of each feature
    """

    # get the data set as a data frame
    (data, target) = fetch_adult(return_X_y=True, as_frame=True)


    print(np.asarray(data.columns))
    print(data.dtypes)

if __name__ == '__main__':
    fetch_adult_data()