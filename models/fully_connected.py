
from typing import Tuple, List
import numpy as np

from ml.models import Model
from ml.modules.initializers import Initializer, RandomInitializer
from ml.modules.activation_functions import linear, softmax, sigmoid

class FullyConnectedANN(Model):
    def __init__(self, 
        input_dim: int,
        layers: List[Tuple[int, str]],
        weight_initializer: Initializer = RandomInitializer(-0.01, 0.01)
    ): 
        self._input_dim = input_dim

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

    def update(self, gradients: np.ndarray, alpha: float) -> None:
        pass

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
        self._weights = weight_initializer(np.zeros((input_dim, output_dim)))
        self._bias = np.zeros((1, output_dim))
        
        if(initialize_bias):
            self._bias = weight_initializer(self._bias)

        if self._activation == 'linear':
            self._activation_fx = linear
        else:
            raise ValueError('Activation function not implemented.')

    def predict(self, X: np.ndarray) -> np.ndarray:
        # make sure dimmenions match.
        assert X.shape[1] == self._weights.shape[0]
        
        return self._activation_fx(np.matmul(X, self._weights) + self._bias)

    def update(self, gradients: np.ndarray, alpha: float) -> None:
        pass

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

