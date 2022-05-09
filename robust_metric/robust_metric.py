import data_util.fetch_data as data_util
from sklearn.svm import SVC
from fairlearn.reductions import DemographicParity, EqualizedOdds


class RobustMetric:
    """
    Holds methods and instances for analysing the robustness of a data set for a certain optimisation problem
    with a particular fairness metric
    """

    def __init__(self, data=None, target=None, sens=None, model_type='SVC', fairness_constraint='DP', max_iter=1000):
        """
        Function to initialise a robustness metric class, which can measure the fairness and robustness of a
        learning method with a specific fairness constraint with a selected data set.
        :param data: The training input dataframe
        :param target: The target output dataframe that corresponds to the input data
        :param sens: The label (as a string) containing the name of the header for the sensitive data column
        :param model_type: Learning model wanted
        :param fairness_constraint: The fairness constraint, which can be handed to the optimisation problem
        :param max_iter: Maximum number of iterations that should be run when training the models
        """

        # if there is no defined data, fetch the adult data set
        if data is None and target is None and sens is None:
            print('No data defined, fetching the ACSIncome data set')
            self.data, self.target, self.sensitive, self.cat, self.bounds = data_util.fetch_adult_data()
        else:
            self.data, self.target, self.sensitive, self.cat, self.bounds = data_util.prepare_data(data, target, sens)

        # define the optimisation method to be used
        self.model_type = 'Support Vector Classification (SVC)'
        model = SVC

        if model_type != 'SVC':
            print('Model not included')

        # define fairness constraint to be used
        self.fairness_constraint = 'Demographic Parity'
        self.fairness_constraint_func = DemographicParity

        if fairness_constraint == 'EO':
            self.fairness_constraint = 'Equalized Odds'
            self.fairness_constraint_func = EqualizedOdds

        # define empty lists for training and testing data across inputs, outputs, and sensitive data
        self.x_tr = []
        self.y_tr = []
        self.sens_tr = []
        self.x_te = []
        self.y_te = []
        self.sens_te = []

        # define variable to hold the different models
        self.baseline_model = model(max_iter=max_iter)

    def problem_summary(self):
        """
        Shows summary of the instance's settings
        :return: Nothing
        """

        print("\n---- SUMMARY OF ROBUSTNESS SETTINGS AND DATA ----\n")
        print('Learning Model: {}'.format(self.model_type))
        print('Fairness Constraint: {}'.format(self.fairness_constraint))
        if len(self.x_tr) == 0:
            print('Data: not split into training and testing batches')
        else:
            print('Data: split into training and testing batches')
        print("\n_________________________________________________")

    def split_data(self, ratio=0.7, seed=666):
        """
        splits the data into a training data set and a testing data set and saves them to the instance
        :param ratio: ratio of the split (eg 0.7 is 70% training, 30% testing)
        :param seed: set pseudo-random seed so experiments can be repeated with same test/train split
        :return: Nothing
        """

        # get name of sensitive feature
        sens_key, sens_value = list(self.sensitive.items())[0]

        self.x_tr, self.y_tr, self.sens_tr, self.x_te, self.y_te, self.sens_te = \
            data_util.split(self.data, self.target, self.sensitive, ratio, seed, sens_name=sens_key)

    def run_baseline(self, max_iter=1000):
        """
        Creates the baseline model
        :param max_iter: maximum number of iterations used to fit the model
        :return: the baseline model and the baseline model score
        """

        # run the model with the training data
        self.baseline_model.fit(self.x_tr, self.y_tr)

        # get score of the baseline model with the testing data
        score = self.baseline_model.score(self.x_te, self.y_te)

        return score
