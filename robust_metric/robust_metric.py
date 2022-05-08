import data_util.fetch_data as data_util


class RobustMetric:
    """
    Holds methods and instances for analysising the robustness of a data set for a certain optimisation problem
    with a particular fairness metric
    """

    def __init__(self):
        self.data, self.target, self.sensitive, self.cat, self.bounds = data_util.fetch_adult_data()


if __name__ == '__main__':
    test = RobustMetric()
