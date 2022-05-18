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
from folktables import ACSDataSource, ACSEmployment, ACSPublicCoverage
from scipy.stats import skewnorm

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
    if args.dataset == 'adult':
        (data, target) = fetch_adult(return_X_y=True, as_frame=True)
        sens = 'sex'

    if args.dataset == 'bank':
        (data, target) = fetch_bank_marketing(return_X_y=True, as_frame=True)

        # sensitive attribute is race
        sens = 'V3'

        # This data set has a bad class ratio, so equalise it
        (data, target) = data_util.fetch_data.equalize_data(data, target)

    if args.dataset == 'boston':
        (data, target) = fetch_boston(return_X_y=True, as_frame=True)
        avg_data = np.mean(data['B'])
        avg_target = np.mean(target)

        # change to binary classifier with binary data
        for i in range(0, len(target)):
            if target[i] < avg_target:
                target[i] = 0
            else:
                target[i] = 1

            if data['B'][i] < avg_data:
                data['B'][i] = 'A'
            else:
                data['B'][i] = 'B'

        sens = 'B'

    if args.dataset == 'compas':
        dataURL = 'https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'
        dfRaw = pd.read_csv(dataURL)

        # check if entire column is NaN. If so, drop it
        dfRaw = dfRaw.dropna(axis=1, how='all')

        # any individual NaN fields fill with mean
        dfRaw.apply(lambda x: x.fillna(x.mode()), axis=0)
        dfRaw = dfRaw.fillna(-1)
        dfRaw = dfRaw.fillna(dfRaw.mode().iloc[0])
        dfRaw = dfRaw.dropna(axis=1)
        data = dfRaw.iloc[:, :-1]
        target = dfRaw.iloc[:, -1]

        data = data.drop(labels='id', axis=1)

        sens = 'race'

    if args.dataset == 'german':
        credit = pd.read_csv('data_input/german.data', header=None, sep=' ')

        # no headers given, so create them
        headers = list(string.ascii_lowercase)
        for i in range(0, len(credit.columns)):
            credit = credit.rename({i: headers[i]}, axis=1)

        # seperate inputs from outputs
        data = credit.iloc[:, :-1]
        target = credit.iloc[:, -1]

        # This data set has a bad class ratio, so equalise it
        (data, target) = data_util.fetch_data.equalize_data(data, target)

        # sensitive attribute is month
        sens = 'i'

    if args.dataset == 'employ':
        cali_data = pd.read_csv('data_input/psam_p06.csv')
        all_data = cali_data[cali_data.columns.intersection(ACSEmployment.features)]
        all_target = cali_data[ACSEmployment.target]
        combine = pd.concat([all_data, all_target], axis=1)
        combine = combine.fillna(0)
        combine = combine.dropna(axis=0)

        data = combine.iloc[:, :-1]
        target = combine.iloc[:, -1]

        target = (target != 1).astype(int)

        sens = 'RAC1P'

    if args.dataset == 'fair':
        size = 20000
        np.random.seed(123)
        data = np.asarray([np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size)])

        data = np.transpose(data)

        target = np.transpose(np.array(np.random.choice([0, 1], size=size)))

        data = pd.DataFrame(data,
                            columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'])

        target = pd.Series(target)

        for col in ['K', 'L', 'M', 'N', 'O', 'P', 'Q']:
            data[col] = data[col].astype('category')

        sens = 'Q'

    if args.dataset == 'unfair':
        size = 20000
        np.random.seed(123)
        data = np.asarray([np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.normal(loc=0.0, scale=1.0, size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size),
                           np.random.choice([0, 1], size=size)])

        data = np.transpose(data)

        target = np.transpose(np.array(np.random.choice([0, 1], size=size)))

        for row in range(0, size):
            for col in range(0, 10):
                if data[row, 16] == 1:
                    data[row, col] = np.random.normal(loc=10, scale=1.0)
                else:
                    data[row, col] = np.random.normal(loc=-10, scale=2.0)
            for col in range(10, 15):
                if data[row, 16] == 1:
                    data[row, col] = np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], replace=True)
                else:
                    data[row, col] = np.random.choice([0, 0, 1, 1, 1, 1, 1, 1, 1, 1], replace=True)

        data = pd.DataFrame(data,
                            columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                     'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q'])

        target = pd.Series(target)

        for col in ['K', 'L', 'M', 'N', 'O', 'P', 'Q']:
            data[col] = data[col].astype('category')

        sens = 'Q'

    levels = np.arange(args.min_noise, args.max_noise + args.min_noise, args.min_noise)

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
    robustness = test.measure_robustness()
    rel_robustness = test.measure_relative_robustness()

    test.summary()

    directory_fairness = '{}/fairness/{}_{}_{}_data'.format(args.output_dir, args.dataset,
                                                            args.model_type, args.train_constraint)

    directory_robustness = '{}/robustness/{}_{}_{}_data'.format(args.output_dir, args.dataset,
                                                                args.model_type, args.train_constraint)

    directory_rel_robustness = '{}/relative_robustness/{}_{}_{}_data'.format(args.output_dir, args.dataset,
                                                                             args.model_type, args.train_constraint)

    np.save(directory_fairness, fairness)
    np.save(directory_robustness, robustness)

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
                        x_lim=[args.min_noise, args.max_noise])

    print('DONE')
