from robust_metric.robust_metric import RobustMetric
from fairlearn.datasets import fetch_adult, fetch_bank_marketing
import numpy as np
import warnings
from matplotlib import pyplot

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    """(data, target) = fetch_bank_marketing(return_X_y=True, as_frame=True)
    test = RobustMetric(data=data, target=target, sens='V9', max_iter=5000, fairness_constraint='dp',
                        noise_level=np.arange(1, 21), noise_iter=10)
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

    np.save('data/fairness_banking_dp_full_nopre', fairness)"""

    test = np.load('data/fairness_banking_dp_full_nopre.npy')

    name = ['Baseline', 'Pre-processing', 'In-processing', 'Post-processing']
    colour1 = ['-r', '-g', '-b', '-k']
    colour2 = ['red', 'green', 'blue', 'black']
    means = np.zeros((4, 21))
    mean_er = np.zeros((4, 21))
    x = np.arange(0, 21)
    for i in range(0, 4):
        for j in range(0, len(means[0])):
            means[i, j] = np.mean(test[i, j])
            mean_er[i, j] = np.std(test[i, j]) / np.sqrt(len(test[i, j]))

    for i in range(0, 4):
        pyplot.plot(np.arange(0, 21), means[i], colour1[i], label=name[i])
        pyplot.fill_between(x, means[i] - mean_er[i], means[i] + mean_er[i], alpha=0.25, color=colour2[i])
    pyplot.show()

    print('\nEnd')
