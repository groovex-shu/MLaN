import numpy as np
import pytest

from lovot_slam.regressor.knn import Regressor


@pytest.mark.parametrize('x_train,y_train,x_query,k,expected', [
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[1, 1], [2, 2], [3, 3]]),
     np.array([1, 2, 3]),  # exactly the same as the training data
     1,
     np.array([1, 1])),
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[1, 1], [2, 2], [3, 3]]),
     np.array([1, 2, 3]),  # exactly the same as the training data
     2,
     np.array([1, 1])),  # NOTE: not exactly [1, 1] because of the epsilon
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[1, 1], [2, 2], [3, 3]]),
     np.array([0, 1, 2]),
     1,
     np.array([1, 1])),
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[1, 1], [2, 2], [3, 3]]),
     np.array([0, 1, 2]),
     2,
     np.array([1.2, 1.2])),  # 1/weight= sqrt(1), sqrt(4**2)
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[1, 1], [2, 2], [3, 3]]),
     np.array([0, 1, 2]),
     3,
     np.array([1.3846153846153848, 1.3846153846153848])),  # 1/weight= sqrt(1), sqrt(4**2), sqrt(7**2)
    (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
     np.array([[1, 1], [2, 2], [3, 3]]),
     np.array([0, 1, 2]),
     4,  # k > len(x_train) -> k = len(x_train)
     np.array([1.3846153846153848, 1.3846153846153848])),  # 1/weight= sqrt(1), sqrt(4**2), sqrt(7**2)
])
def test_predict(x_train, y_train, x_query, k, expected):
    regressor = Regressor(x_train, y_train)
    actual = regressor.predict(x_query, k)
    assert np.isclose(actual, expected, atol=1e-6).all()


def test_predict_assertion():
    regressor = Regressor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[1, 1], [2, 2], [3, 3]]))
    with pytest.raises(AssertionError):
        _ = regressor.predict(np.array([1, 2, 3]), 0)
