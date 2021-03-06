import argparse
import numpy as np
import os
import warnings
import data_util.fetch_data
import data_util.plot_data as plot_data
import data_util.ROC as roc
from robust_metric.RobustMetric import RobustMetric

# Remove warnings from printed output
warnings.filterwarnings("ignore")

# Define the arguments that can be taken to change to type of analysis
parser = argparse.ArgumentParser(description="evaluate the robustness of models")
parser.add_argument('--dataset', type=str, default='adult',
                    help='select dataset to test')
parser.add_argument('--train_constraint', type=str, default='dp',
                    help='using which constraint to train the model, including eo, dp, fp, tp')
parser.add_argument('--output_dir', type=str, default='test', help='output dir for saving the result')
parser.add_argument('--min_noise', type=float, default=0.01, help='minimum level of noise for test')
parser.add_argument('--max_noise', type=float, default=1, help='maximum level of noise for test')
parser.add_argument('--noise_iters', type=int, default=10, help='Number of data samples per noise level')
parser.add_argument('--model_iters', type=int, default=1000, help='Maximum iterations for model fitting')
parser.add_argument('--model_type', type=str, default='SVC', help='Type of model to optimise, '
                                                                  'including SVC, MLP, LR, SGD, DTC')
parser.add_argument('--step_size', type=float, default=1, help='Step size for noise levels')
parser.add_argument('--roc', type=bool, default=False, help='Plot and save the ROC curve for k=0 and k=1')
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
    if not os.path.exists(args.output_dir + '/fairness'):
        os.makedirs(args.output_dir + '/fairness')

    if not os.path.exists(args.output_dir + '/robustness'):
        os.makedirs(args.output_dir + '/robustness')

    if not os.path.exists(args.output_dir + '/relative_robustness'):
        os.makedirs(args.output_dir + '/relative_robustness')

    # set up variables from arguments
    data, target, sens = data_util.fetch_data.get_data(args.dataset)

    levels = np.arange(args.min_noise, args.max_noise + args.min_noise, args.step_size)

    test = RobustMetric(data=data, target=target, sens=sens, max_iter=args.model_iters, model_type=args.model_type,
                        fairness_constraint=args.train_constraint, noise_level=levels,
                        noise_iter=args.noise_iters)
    test.split_data()

    # Train models with baseline, pre and post processing, and in-learning
    score_base = test.run_baseline()
    score_pre = test.run_preprocessing()
    score_in = test.run_inprocessing()
    score_post = test.run_postprocessing()

    # Measure fairness and robustness metrics
    fairness = test.measure_total_fairness()
    robustness = test.measure_robustness()
    rel_robustness = test.measure_relative_robustness()

    # Save all the data
    directory_fairness = '{}/fairness/{}_{}_{}_data'.format(args.output_dir, args.dataset,
                                                            args.model_type, args.train_constraint)

    directory_robustness = '{}/robustness/{}_{}_{}_data'.format(args.output_dir, args.dataset,
                                                                args.model_type, args.train_constraint)

    directory_rel_robustness = '{}/relative_robustness/{}_{}_{}_data'.format(args.output_dir, args.dataset,
                                                                             args.model_type, args.train_constraint)

    np.save(directory_fairness, fairness)
    np.save(directory_robustness, robustness)

    # plot and save random noise fairness and robustness
    plot_data.plot_data(fairness, levels, directory_fairness + '_fairness_figure', save=True,
                        title='Fairness of {} with {}'.format(args.dataset, test.model_type),
                        x_label='Noise Level', y_label=full_constraints[args.train_constraint])

    plot_data.plot_data(robustness, levels, directory_robustness + '_robustness_figure', save=True,
                        title='Robustness of {} with {}'.format(args.dataset, test.model_type),
                        x_label='Noise Level', y_label=full_constraints[args.train_constraint],
                        x_lim=[args.min_noise, args.max_noise])

    plot_data.plot_data(rel_robustness, levels, directory_rel_robustness + '_rel_robustness_figure', save=True,
                        title='Relative Robustness of {} with {}'.format(args.dataset, test.model_type),
                        x_label='Noise Level', y_label=full_constraints[args.train_constraint],
                        x_lim=[args.min_noise, args.max_noise], log=True)

    print('Baseline accuracy score: ' + str(score_base))
    print('Pre-processing accuracy score: ' + str(score_pre))
    print('In-processing accuracy score: ' + str(score_in))
    print('Post-processing accuracy score: ' + str(score_post))
    test.summary()

    if args.roc:
        roc.plot_ROC(test, directory_rel_robustness)

    print('DONE')
