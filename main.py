from robust_metric.robust_metric import RobustMetric
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    test = RobustMetric()
    test.problem_summary()
    test.split_data()
    test.problem_summary()
    score = test.run_baseline()

    print(score)
