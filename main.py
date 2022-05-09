from robust_metric.robust_metric import RobustMetric
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    test = RobustMetric()
    test.split_data()
    score_base = test.run_baseline()
    score_in = test.run_inprocessing()

    print(score_base)
    print(score_in)
