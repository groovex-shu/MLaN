from logging import getLogger

import numpy as np

_logger = getLogger(__name__)


class Regressor:
    """
    Simple implementation of kNN regressor.
    """

    def __init__(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        :param x_train: training data (n*m array: n_samples, m_features)
        :param y_train: training label (n*o array: n_samples, o_outputs)
        """
        self._x_train = x_train
        self._y_train = y_train

    def predict(self, x_query: np.ndarray, k: int) -> np.ndarray:
        """
        Predict the label of the query data.
        Accept only one query data at a time.
        TODO: accept multiple query data at a time.

        :param x_query: query data (m array: m_features)
        :param k: number of nearest neighbors
        :return: predicted label (o array: o_outputs)
        """
        assert k > 0, 'k must be positive'

        def dist(x_train, x_query) -> float:
            return np.sqrt(((x_train - x_query) ** 2).sum())

        distances = np.zeros((len(self._x_train)))

        for i in range(len(self._x_train)):
            distances[i] = dist(self._x_train[i], x_query)

        k_indexes = np.argsort(distances)[:k]
        epsilon = 1e-6  # avoid zero division
        estimation = np.average(self._y_train[k_indexes], axis=0,
                                weights=1 / (distances[k_indexes] + epsilon))
        return estimation
