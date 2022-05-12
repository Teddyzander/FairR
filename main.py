import argparse
import numpy as np
import os
import warnings
import data_util.plot_data as plot_data
from robust_metric.RobustMetric import RobustMetric
from fairlearn.datasets import fetch_adult, fetch_bank_marketing

# Remove warnings from printed output
warnings.filterwarnings("ignore")

# Define the arguments that can be taken to change to type of analysis
parser = argparse.ArgumentParser(description="evaluate the robustness of models")
parser.add_argument('--dataset', type=str, default='adult',
                    help='select dataset to test')
parser.add_argument('--train_constraint', type=str, default='dp',
                    help='using which constraint to train the model, including eo, dp, fp, tp')
parser.add_argument('--output_dir', type=str, default='data', help='output dir for saving the result')
parser.add_argument('--max_noise', type=int, default=20, help='maximum level of noise for test')
parser.add_argument('--noise_iters', type=int, default=10, help='Number of data samples per noise level')
parser.add_argument('--model_iters', type=int, default=5000, help='Maximum iterations for model fitting')
args = parser.parse_args()

# Dictionary to hold full titles of training constraints (used for plot axis)
full_constraints = {'dp': 'Demographic Parity',
                    'eo': 'Equalized Odds',
                    'fp': 'False Positive',
                    'tp': 'True Positive'}

if __name__ == '__main__':

    # Print summary of analysis
    print('Fairness Constraint: {}\n'
          'Data-set:  {}\n'
          'Maximum Noise: {}\n'
          'Iterations per Noise Level: {}\n'
          'Iterations to Fit Models: {}\n'
          .format(full_constraints[args.train_constraint], args.dataset,
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

    test = RobustMetric(data=data, target=target, sens=sens, max_iter=args.model_iters,
                        fairness_constraint=args.train_constraint, noise_level=np.arange(1, args.max_noise + 1),
                        noise_iter=args.noise_iters)
    test.split_data()

    score_base = test.run_baseline()
    score_pre = test.run_preprocessing()
    score_in = test.run_inprocessing()
    score_post = test.run_postprocessing()

    fairness = test.measure_total_fairness()

    print('Baseline accuracy score: ' + str(score_base))
    print('Pre-processing accuracy score: ' + str(score_pre))
    print('In-processing accuracy score: ' + str(score_in))
    print('Post-processing accuracy score: ' + str(score_post))

    test.summary()

    directory = '{}/fairness_{}_{}_data'.format(args.output_dir, args.dataset, args.train_constraint)

    np.save(directory, fairness)

    test = np.load(directory + '.npy')

    plot_data.plot_data(test, np.arange(1, args.max_noise + 1), directory + '_figure', save=True,
                        x_label='Noise Level', y_label=full_constraints[args.train_constraint])
