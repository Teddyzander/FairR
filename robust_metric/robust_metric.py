import data_util.fetch_data as data_util
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics._classification import accuracy_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.preprocessing import CorrelationRemover
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient


class RobustMetric:
    """
    Holds methods and instances for analysing the robustness of a data set for a certain optimisation problem
    with a particular fairness metric
    """

    def __init__(self, data=None, target=None, sens='sex', model_type='SVC',
                 fairness_constraint='dp', max_iter=1000):
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

        # save number of maximum number of iterations for optimisation problems
        self.max_iter = max_iter

        # if there is no defined data, fetch the adult data set
        if data is None and target is None:
            print('No data defined, fetching the ACSIncome data set')
            self.data, self.target, self.sensitive, self.cat, self.bounds = data_util.fetch_adult_data(sens=sens)
        else:
            self.data, self.target, self.sensitive, self.cat, self.bounds = data_util.prepare_data(data, target, sens)

        # define the optimisation method to be used
        self.model_type = 'Support Vector Classification (SVC)'
        self.model = SVC

        if model_type != 'SVC':
            print('Model not included')

        # define fairness constraint to be used
        self.fairness_constraint = 'demographic_parity'
        self.fairness_constraint_full = 'Demographic Parity'
        self.fairness_constraint_func = DemographicParity()

        if fairness_constraint == 'eo':
            self.fairness_constraint = 'equalized_odds'
            self.fairness_constraint_full = 'Equalized Odds'
            self.fairness_constraint_func = EqualizedOdds()

        # define empty lists for training and testing data across inputs, outputs, and sensitive data
        self.x_tr = []
        self.y_tr = []
        self.sens_tr = []
        self.x_te = []
        self.y_te = []
        self.sens_te = []

        # define variable to hold the different models
        self.baseline_model = None
        self.inprocessing_model = None
        self.preprocessing_model = None
        self.postprocessing_model = None

    def summary(self):
        """
        Shows summary of the instance's settings
        :return: Nothing
        """

        print('\n---- SUMMARY OF ROBUSTNESS SETTINGS AND DATA ----\n')

        # Check settings for the class instance
        print('Learning Model: {}'.format(self.model_type))
        print('Fairness Constraint: {}'.format(self.fairness_constraint_full))
        print('Sensitive Feature: {}'.format(list(self.sensitive.keys())[0]))
        print('Maximum number of iterations: {}'.format(self.max_iter))

        # Check to see if the data has been separated into training and testing
        if len(self.x_tr) == 0:
            print('Data: not split into training and testing batches')
        else:
            print('Data: split into training and testing batches')

        # Check if models exist
        print('Baseline Model: {}'.format(self.baseline_model))
        print('Pre-processing Model: {}'.format(CorrelationRemover) + ' with {}'.format(self.preprocessing_model))
        print('In-processing Model: {}'.format(self.inprocessing_model))
        print('Post-processing Model: {}'.format(self.postprocessing_model))

        print('\n_________________________________________________')

    def change_max_iter(self, new_max_iter):
        """
        Change the maximum number of iterations for the optimisation problems
        :param new_max_iter: new max iterations setting
        :return: Nothing
        """
        self.max_iter = new_max_iter

    def split_data(self, ratio=0.7, seed=123):
        """
        splits the data into a training data set and a testing data set and saves them to the instance
        :param ratio: ratio of the split (eg 0.7 is 70% training, 30% testing)
        :param seed: set pseudo-random seed so experiments can be repeated with same test/train split
        :return: Nothing
        """

        # get name of sensitive feature
        sens_key = list(self.sensitive.keys())[0]

        self.x_tr, self.y_tr, self.sens_tr, self.x_te, self.y_te, self.sens_te = \
            data_util.split(self.data, self.target, self.sensitive, ratio, seed, sens_name=sens_key)

    def run_baseline(self):
        """
        Creates the baseline model
        :return: The baseline score, which represents the prediction accuracy on the testing data
        """

        print('Fitting baseline model...')

        # run the model with the training data
        self.baseline_model = self.model(max_iter=self.max_iter)
        self.baseline_model.fit(self.x_tr, self.y_tr)

        # get score of the baseline model with the testing data
        score = self.baseline_model.score(self.x_te, self.y_te)

        return score

    def run_preprocessing(self):
        """
        Creates the preprocessing model by applying a linear transformation on the data to remove the correlation
        to the sensitive features
        :return: The preprocessing score, which represents the prediction accuracy on the testing data
        """

        print('Fitting pre-processing model...')

        # Before removing the correlation, we need to stitch the data back together so that we have access to the
        # sensitive data (which is stored as the last column)
        x_tr_pre = np.concatenate([self.x_tr, self.sens_tr.reshape(-1, 1)], axis=1)
        x_te_pre = np.concatenate([self.x_te, self.sens_te.reshape(-1, 1)], axis=1)

        # Now we can find a linear transformation that removes the correlation to the sensitive features
        remover = CorrelationRemover(sensitive_feature_ids=[x_tr_pre.shape[1] - 1])
        remover.fit(x_tr_pre, self.y_tr)

        # Apply the transformation to the training and testing data
        x_tr_pre = remover.transform(x_tr_pre)
        x_te_pre = remover.transform(x_te_pre)

        # fit the model on the preprocessed data
        self.preprocessing_model = self.model(max_iter=self.max_iter)
        self.preprocessing_model.fit(x_tr_pre, self.y_tr)

        # get score of the pre-processing model with the testing data
        score = self.baseline_model.score(x_te_pre, self.y_te)

        return score

    def run_inprocessing(self, eps=0.01, nu=1e-6, random_state=123):
        """
        Run the in-processing optimisation with a fairness constraint
        :param random_state: psuedo-random seed to repeat experiments
        :param eps: Float for the allowed fairness violation
        :param nu: Float for convergence threshold
        :return: The in-processing score, which represents the prediction accuracy on the testing data
        """

        print('Fitting in-processing model...')

        # define the model
        self.inprocessing_model = ExponentiatedGradient(self.model(max_iter=self.max_iter),
                                                        constraints=self.fairness_constraint_func,
                                                        eps=eps, nu=nu)

        # run the optimisation model for the defined parameters over the training dataset
        self.inprocessing_model.fit(self.x_tr, self.y_tr, sensitive_features=self.sens_tr)

        # get score of the in-processing model with the testing data
        output = self.inprocessing_model.predict(self.x_te, random_state=random_state)
        score = 1 - np.mean(np.abs(output - self.y_te))

        return score

    def run_postprocessing(self):
        """
        Runs the post-processing model, adjusting the threshold of the baseline model in order to optimise for the
        fairness constraint. Must run baseline first
        :return: The post-processing score, which represents the prediction accuracy on the testing data
        """

        print('Fitting post-processing model...')

        # define post-processing model
        self.postprocessing_model = ThresholdOptimizer(
            estimator=self.baseline_model,
            constraints=self.fairness_constraint,
            prefit=True)

        # fit the postprocessing model with the allocated fairness constraint
        self.postprocessing_model.fit(self.x_tr, self.y_tr, sensitive_features=self.sens_tr)

        # get score of the post-processing model with the testing data
        output = self.postprocessing_model.predict(self.x_te, sensitive_features=self.sens_te)
        score = 1 - np.mean(np.abs(output - self.y_te))

        return score
