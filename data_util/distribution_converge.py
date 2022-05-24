import numpy as np
import pandas as pd
import time
import data_util.fetch_data


def con_dist(fairness_obj, levels, name=None):
    """
    Tests for fairness behaviour as distributions converge towards being completely fair
    :param fairness_obj: Trained robustness model
    :param levels: convergence levels for t=(0, 1) with 0 being no convergence and 1 being total convergence
    :param name: data set being used
    :return: fairness measures across different processing methods for reduced unfairness in data
    """
    fairness = None
    if name == 'unfair_1':
        level = 0
        fairness = np.zeros((4, len(levels)))
        for t in levels:
            start = time.time()
            size = 500000
            np.random.seed(123)
            data = np.asarray([np.random.normal(loc=0.0, scale=1.0, size=size),
                               np.random.normal(loc=0.0, scale=1.0, size=size),
                               np.random.choice([-1, 1], size=size)])

            data = np.transpose(data)

            target = np.transpose(np.array(np.random.choice([-1, 1], size=size)))

            for row in range(0, size):
                data[row, 0] = np.random.choice([data[row, 2], -data[row, 2]], p=[1-(t/2), t/2])
                prob = 1 / (1 + np.exp(-2 * data[row, 2] * (-1)))
                temp = 0.5 - prob
                target[row] = np.random.choice([-1, 1], p=[prob + t * temp, 1 - prob - t * temp])
                data[row, 1] = target[row] + np.random.normal(loc=0, scale=1.0)

            data = pd.DataFrame(data,
                                columns=['A', 'B', 'C'])

            target = pd.Series(target)

            for col in ['C']:
                data[col] = data[col].astype('category')

            sens = 'C'
            data, target, sense, cat, bounds = data_util.fetch_data.prepare_data(data, target, sens)
            fairness[0, level], fairness[1, level], fairness[2, level], fairness[3, level], = \
                fairness_obj.measure_fairness(data, target, sense[sens])

            end = time.time()

            if level == 0:
                completion_est = np.round(((len(levels) * (end - start + 0.01)) / 60),
                                          decimals=2)
                print('Measuring fairness for distribution convergence. '
                      'Estimated time to completion: {} minutes from {}'
                      .format(completion_est, time.strftime("%H:%M:%S")))

            level += 1



    if name == 'unfair_2':
        print('Testing fairness for distribution convergence...')
        fairness = np.zeros((4, len(levels)))
        level = 0
        mean1 = (2 * np.exp(2)) / (np.exp(2) + 1)
        mean0 = 0
        var1 = (np.exp(4) - 1) / ((np.exp(2) + 1) ** 2)
        var0 = (np.exp(-4) + 1) / ((np.exp(-2) + 1) ** 2)
        for t in levels:
            start = time.time()
            size = 500000
            np.random.seed(123)
            data = np.asarray([np.random.normal(loc=0.0, scale=1.0, size=size),
                               np.random.normal(loc=0.0, scale=1.0, size=size),
                               np.random.normal(loc=0.0, scale=1.0, size=size),
                               np.random.choice([-1, 1], size=size)])

            data = np.transpose(data)

            target = np.transpose(np.array(np.random.choice([-1, 1], size=size)))

            for row in range(0, size):
                if data[row, 3] == 1:
                    data[row, 0] = np.random.choice([-1, 1], p=[t / 2, 1 - (t / 2)])
                    data[row, 2] = np.random.normal(loc=mean1, scale=var1)
                else:
                    data[row, 0] = np.random.choice([-1, 1], p=[1 - (t / 2), t / 2])
                    data[row, 2] = np.random.normal(loc=(1 - t) * mean0 + t * mean1, scale=(1 - t) * var0 + t * var1)
                prob = 1 / (1 + np.exp(-2 * data[row, 2]))
                temp = 0.5 - prob
                target[row] = np.random.choice([-1, 1], p=[1 - prob - t * temp, prob + t * temp])
                data[row, 1] = target[row] + np.random.normal(loc=0, scale=1.0)

            data = pd.DataFrame(data,
                                columns=['A', 'B', 'C', 'D'])

            target = pd.Series(target)

            for col in ['D']:
                data[col] = data[col].astype('category')

            sens = 'D'

            data, target, sense, cat, bounds = data_util.fetch_data.prepare_data(data, target, sens)
            fairness[0, level], fairness[1, level], fairness[2, level], fairness[3, level], = \
                fairness_obj.measure_fairness(data, target, sense[sens])

            end = time.time()

            if level == 0:
                completion_est = np.round(((len(levels) * (end - start + 0.01)) / 60),
                                          decimals=2)
                print('Testing fairness for distribution convergence. Estimated time to completion: {} minutes from {}'
                      .format(completion_est, time.strftime("%H:%M:%S")))

            level += 1

    return fairness
