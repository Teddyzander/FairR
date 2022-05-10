from robust_metric.robust_metric import RobustMetric
from fairlearn.datasets import fetch_adult
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    (data, target) = fetch_adult(return_X_y=True, as_frame=True)
    test = RobustMetric(data=data, target=target, sens='sex', max_iter=10000, fairness_constraint='dp',
                        noise_level=[0.1, 0.5, 1, 2], noise_iter=10)
    test.split_data()

    test.gen_noise()

    score_base = test.run_baseline()
    score_pre = test.run_preprocessing()
    score_in = test.run_inprocessing()
    score_post = test.run_postprocessing()

    fairness = test.measure_fairness()

    print('Baseline accuracy score: ' + str(score_base))
    print('Pre-processing accuracy score: ' + str(score_pre))
    print('In-processing accuracy score: ' + str(score_in))
    print('Post-processing accuracy score: ' + str(score_post))

    test.summary()

    print('\nEnd')
