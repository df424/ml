
from typing import Callable, Dict, Any
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
        self._activation = activation
        self._regularizer = regularizer
        self._input_cache = {}
        self._parameters = {}
        self._gradients = {}

        self._parameters['W'] = weight_initializer.initialize(np.zeros((self._input_dim, self._output_dim)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        W = self._parameters['W']
        y_hat = self._activation(np.matmul(X, W))
        self._input_cache['X'] = X
        return y_hat

    def backward(self, loss: np.ndarray) -> None:
        X = self._input_cache['X']
        self._gradients['W'] = np.matmul(X.T, loss)/X.shape[0]
        # self._weights = self._weights - alpha * 
        # if self._regularizer:
        #     self._regularizer.regularize(self._weights)

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def gradients(self) -> Dict[str, Any]:
        return self._gradients
        

