import data_util.fetch_data as data_util
from sklearn.svm import SVC
from fairlearn.reductions import DemographicParity


class RobustMetric:
    """
    Holds methods and instances for analysing the robustness of a data set for a certain optimisation problem
    with a particular fairness metric
    """

    def __init__(self, data=None, target=None, sens=None, model='SVC', fairness_constraint='DP'):
        """
        Function to initialise a robustness metric class, which can measure the fairness and robustness of a
        learning method with a specific fairness constraint with a selected data set.
        :param data: The training input dataframe
        :param target: The target output dataframe that corresponds to the input data
        :param sens: The label (as a string) containing the name of the header for the sensitive data column
        :param model: Learning model wanted
        :param fairness_constraint: The fairness constraint, which can be handed to the optimisation problem
        """
        if data is None and target is None and sens is None:
            self.data, self.target, self.sensitive, self.cat, self.bounds = data_util.fetch_adult_data()
        else:
            self.data, self.target, self.sensitive, self.cat, self.bounds = data_util.prepare_data(data, target, sens)

        if model == 'SVC':
            self.model = 'Support Vector Classification (SVC)'
            self.train_model = SVC

        if fairness_constraint == 'DP':
            self.fairness_constraint = 'Demographic Parity'
            self.fairness_constraint_func = DemographicParity

        self.x_tr = []
        self.y_tr = []
        self.sens_tr = []
        self.x_te = []
        self.y_te = []
        self.sens_te = []

    def problem_summary(self):
        """
        Shows summary of the instance's settings
        :return:
        """
        print('Learning model: ' + self.model)
        print('Fairness constraint: ' + self.fairness_constraint)
        if len(self.x_tr) == 0:
            print('Data is not split')
        else:
            print('Data is split')

    def split_data(self, ratio=0.7, seed=666):

        sens_key, sens_value = list(self.sensitive.items())[0]
        self.x_tr, self.y_tr, self.sens_tr, self.x_te, self.y_te, self.sens_te = \
            data_util.split(self.data, self.target, self.sensitive, ratio, seed, sens_name=sens_key)

    def run_baseline(self):
        """
        Creates the baseline model
        :return:
        """
