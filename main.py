from robust_metric.robust_metric import RobustMetric
from fairlearn.datasets import fetch_adult
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    (data, target) = fetch_adult(return_X_y=True, as_frame=True)
    test = RobustMetric(data=data, target=target, sens='sex', max_iter=5, fairness_constraint='dp')
    test.split_data()
    score_base = test.run_baseline()
    score_pre = test.run_preprocessing()
    score_in = test.run_inprocessing()
    score_post = test.run_postprocessing()

    print('Baseline score: ' + str(score_base))
    print('Pre-processing score: ' + str(score_pre))
    print('In-processing score: ' + str(score_in))
    print('Post-processing score: ' + str(score_post))

    test.summary()
