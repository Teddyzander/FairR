import numpy as np
import warnings
import data_util.plot_data as plot_data
from robust_metric.robust_metric import RobustMetric
from fairlearn.datasets import fetch_adult, fetch_bank_marketing
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

    plot_data.plot_data(test, 'fairness_banking_dp', save=True)

