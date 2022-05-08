import data_util.fetch_data as data_util


class RobustMetric:
    """
    Holds methods and instances for analysising the robustness of a data set for a certain optimisation problem
    with a particular fairness metric
    """

    def __init__(self, data=None, target=None, sens=None):
        if data is None and target is None and sens is None:
            print("fetching adult data...")
            self.data, self.target, self.sensitive, self.cat, self.bounds = data_util.fetch_adult_data()
        else:
            print("processing data inputted data...")
            self.data, self.target, self.sensitive, self.cat, self.bounds = data_util.prepare_data(data, target, sens)
