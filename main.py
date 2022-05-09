from robust_metric.robust_metric import RobustMetric
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    test = RobustMetric(max_iter=1000)
    test.split_data()
    test.problem_summary()
    score_base = test.run_baseline()
    # score_in = test.run_inprocessing()
    score_pre = test.run_preprocessing()

    print(score_base)
    print(score_pre)

    test.problem_summary()
