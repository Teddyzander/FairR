import data_util.fetch_data as data_util

class RobustMetric:
    """
    Holds methods and instances for analysising the robustness of a data set for a certain optimisation problem
    with a particular fairness metric
    """

    def __int__(self, fair_metric, data=None, model="SVC"):
        if data is None:
            self.data = data_util.fetch_adult_data()
        else:
            self.data = data_util.standardise_data()
