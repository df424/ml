
from typing import Callable
from ml.models import Model
from ml.modules.initializers import Initializer
from ml.modules.regularizers import Regularizer
import numpy as np

class LogisticRegressionClassifier(Model):
    def __init__(self, 
        input_dim: int,
        output_dim: int,
        weight_initializer: Initializer,
        activation: Callable[[np.ndarray], float],
        regularizer: Regularizer = None
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._weights = weight_initializer.initialize(np.zeros((self._input_dim, self._output_dim)))
        self._activation = activation
        self._regularizer = regularizer
        self._input_cache = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_hat = self._activation(np.matmul(X, self._weights))
        self._input_cache['X'] = X
        return y_hat

    def backward(self, loss: np.ndarray, alpha: float) -> None:
        X = self._input_cache['X']
        self._weights = self._weights - alpha * np.matmul(X.T, loss)/X.shape[0]
        if self._regularizer:
            self._regularizer.regularize(self._weights)
