import data_util.fetch_data as data_util

class RobustMetric:
    """
    Holds methods and instances for analysising the robustness of a data set for a certain optimisation problem
    with a particular fairness metric
    """

    def __int__(self, fair_metric, sens, data=None, target=None, model="SVC"):
        if data is None:
            self.data = data_util.fetch_adult_data()
        else:
            self.data, self.target, self. = data_util.prepare_data(data, target, sens)

        self.
