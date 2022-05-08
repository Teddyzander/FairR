from robust_metric.robust_metric import RobustMetric
from fairlearn.datasets import fetch_adult
import data_util.fetch_data as data_util

if __name__ == '__main__':
    test = RobustMetric()
    print(test.cat)

    (data, target) = fetch_adult(return_X_y=True, as_frame=True)
    sens = 'sex'
    test2 = RobustMetric(data, target, sens)
    print(test.cat)

