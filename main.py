import argparse
import numpy as np
import os
import pandas as pd
import string
import warnings
import data_util.fetch_data
import data_util.plot_data as plot_data
from robust_metric.RobustMetric import RobustMetric
from fairlearn.datasets import fetch_adult, fetch_bank_marketing, fetch_boston

# Remove warnings from printed output
warnings.filterwarnings("ignore")

# Define the arguments that can be taken to change to type of analysis
parser = argparse.ArgumentParser(description="evaluate the robustness of models")
parser.add_argument('--dataset', type=str, default='adult',
                    help='select dataset to test')
parser.add_argument('--train_constraint', type=str, default='dp',
                    help='using which constraint to train the model, including eo, dp, fp, tp')
parser.add_argument('--output_dir', type=str, default='test', help='output dir for saving the result')
parser.add_argument('--max_noise', type=float, default=20, help='maximum level of noise for test')
parser.add_argument('--noise_iters', type=int, default=10, help='Number of data samples per noise level')
parser.add_argument('--model_iters', type=int, default=1000, help='Maximum iterations for model fitting')
parser.add_argument('--model_type', type=str, default='SVC', help='Type of model to optimise, '
                                                                  'including SVC, MLP, LR, SGD, DTC')
args = parser.parse_args()

# Dictionary to hold full titles of training constraints (used for plot axis)
full_constraints = {'dp': 'Demographic Parity',
                    'eo': 'Equalized Odds',
                    'fp': 'False Positive',
                    'tp': 'True Positive'}

if __name__ == '__main__':

    # Print summary of analysis
    print('Fairness Constraint: {}\n'
          'Model Type: {}\n'
          'Data-set:  {}\n'
          'Maximum Noise: {}\n'
          'Iterations per Noise Level: {}\n'
          'Iterations to Fit Models: {}\n'
          .format(full_constraints[args.train_constraint], args.model_type, args.dataset,
                  args.max_noise, args.noise_iters, args.model_iters))

    # if the specified directory doesn't exist, we need to create it
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # set up variables from arguments
    if args.dataset == 'adult':
        (data, target) = fetch_adult(return_X_y=True, as_frame=True)
        sens = 'sex'

    if args.dataset == 'bank':
        (data, target) = fetch_bank_marketing(return_X_y=True, as_frame=True)
        sens = 'V3'

        # This data set has a bad class ratio, so equalise it
        (data, target) = data_util.fetch_data.equalize_data(data, target)

    if args.dataset == 'german':
        credit = pd.read_csv('data_input/german.data', header=None, sep=' ')

        # no headers given, so create them
        headers = list(string.ascii_lowercase)
        for i in range(0, len(credit.columns)):
            credit = credit.rename({i: headers[i]}, axis=1)

        # seperate inputs from outputs
        data = credit.iloc[:, :-1]
        target = credit.iloc[:, -1]
        print(data.dtypes)
        sens = 'i'

    levels = np.arange(0.05, args.max_noise + 0.05, 0.05)

    test = RobustMetric(data=data, target=target, sens=sens, max_iter=args.model_iters, model_type=args.model_type,
                        fairness_constraint=args.train_constraint, noise_level=levels,
                        noise_iter=args.noise_iters)
    test.split_data()

    score_base = test.run_baseline()
    print('Baseline accuracy score: ' + str(score_base))
    score_pre = test.run_preprocessing()
    print('Pre-processing accuracy score: ' + str(score_pre))
    score_in = test.run_inprocessing()
    print('In-processing accuracy score: ' + str(score_in))
    score_post = test.run_postprocessing()
    print('Post-processing accuracy score: ' + str(score_post))

    fairness = test.measure_total_fairness()

    test.summary()

    directory = '{}/fairness_{}_{}_{}_data'.format(args.output_dir, args.dataset,
                                                   args.model_type, args.train_constraint)

    np.save(directory, fairness)

    test_data = np.load(directory + '.npy')

    plot_data.plot_data(test_data, levels, directory + '_figure', save=True,
                        title='{} dataset with {}'.format(args.dataset, test.model_type),
                        x_label='Noise Level', y_label=full_constraints[args.train_constraint])
