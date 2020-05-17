
import numpy as np

from ml.modules.regularizers import Regularizer

class L2Regularizer(Regularizer):
    def __init__(self, lmbda: float, ignore_bias: bool=True):
        self._lambda = lmbda
        self._ignore_bias = ignore_bias

    def regularize(self, theta: np.ndarray) -> None:
        start_idx = 1 if self._ignore_bias else 0
        theta[start_idx:, :] -= (theta[start_idx:, :] * self._lambda)