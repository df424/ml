
from typing import Tuple, List
import numpy as np

from ml.models import Model
from ml.modules.initializers import Initializer, RandomInitializer
from ml.modules.activation_functions import linear, softmax, sigmoid
from ml.modules.regularizers import Regularizer

class FullyConnectedANN(Model):
    def __init__(self, 
        input_dim: int,
        layers: List[Tuple[int, str]],
        weight_initializer: Initializer = RandomInitializer(-0.01, 0.01),
        regularizer: Regularizer = None
    ): 
        self._input_dim = input_dim
        self._regularizer = regularizer

        assert len(layers) > 0

        self._layers = []

        for i, l in enumerate(layers):
            if i == 0: # if its the first layer...
                layer_input_dim=self._input_dim
            else: # Must not be the first layer.
                layer_input_dim=self._layers[-1].output_dim

            self._layers.append(
                FullyConnectedLayer(
                    input_dim=layer_input_dim,
                    output_dim=l[0],
                    weight_initializer=weight_initializer,
                    activation=l[1]
                ))

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = X
        for l in self._layers:
            h = l.predict(h)
        return h

    def backward(self, loss: np.ndarray, alpha: float) -> None:
        dl = loss
        for l in reversed(self._layers):
            dl = l.backward(dl, alpha)
        
        for l in self._layers:
            l.update_weights(alpha, self._regularizer)

class FullyConnectedLayer(Model):
    def __init__(self, 
        input_dim: int,
        output_dim: int,
        weight_initializer: Initializer,
        activation: str = 'relu',
        initialize_bias: bool = False
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._activation = activation
        self._weights = weight_initializer.initialize(np.zeros((input_dim, output_dim)))
        self._bias = np.zeros((1, output_dim))
        self._forward_cache = {}
        self._backward_cache = {}
        
        if(initialize_bias):
            self._bias = weight_initializer.initialize(self._bias)

        if self._activation == 'linear':
            self._activation_fx = linear
        elif self._activation == 'sigmoid':
            self._activation_fx = sigmoid
        else:
            raise ValueError('Activation function not implemented.')

    def predict(self, X: np.ndarray) -> np.ndarray:
        # make sure dimmenions match.
        assert X.shape[1] == self._weights.shape[0]
        #print(X.shape, self._weights.shape, self._bias.shape) 
        Z = np.matmul(X, self._weights) + self._bias
        A = self._activation_fx(Z)
        self._forward_cache['A_prev'] = X
        self._forward_cache['Z'] = Z 
        self._forward_cache['A'] = A
        return A

    def _get_activation_derivative(self, a):
        if self._activation == 'linear':
            return np.ones(a.shape)
        if self._activation == 'sigmoid':
            return a*(1 - a)

    def backward(self, dL: np.ndarray, alpha: float) -> np.ndarray:
        A_prev = self._forward_cache['A_prev']
        Z = self._forward_cache['Z']
        A = self._forward_cache['A']
        dz = dL * self._get_activation_derivative(A)
        dw = (np.matmul(A_prev.T, dz))/A_prev.shape[0]
        db = np.mean(dz, axis=0, keepdims=True)

        # Cache the gradients for use by the optimizer.
        self._backward_cache['dw'] = dw
        self._backward_cache['db'] = db
 
        loss_output = np.matmul(dz, self._weights.T)
        return loss_output

    def update_weights(self, alpha: float, regularizer: Regularizer):
        self._weights -= alpha * self._backward_cache['dw']
        if(regularizer):
            regularizer.regularize(self._weights)
        self._bias -= alpha * self._backward_cache['db']

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

