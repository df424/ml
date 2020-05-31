
from typing import Tuple, List, Dict, Any
import numpy as np

from ml.models import Model
from ml.models.layered_model import LayeredModel
from ml.modules.initializers import Initializer, RandomInitializer
from ml.modules.activation_functions import linear, softmax, sigmoid
from ml.modules.regularizers import Regularizer

class FullyConnectedANN(LayeredModel):
    def __init__(self, 
        input_dim: int,
        layers: List[Tuple[int, str]],
        weight_initializer: Initializer = RandomInitializer(-0.01, 0.01),
        regularizer: Regularizer = None
    ): 
        super(FullyConnectedANN, self).__init__()
        self._input_dim = input_dim
        self._regularizer = regularizer

        assert len(layers) > 0

        for i, l in enumerate(layers):
            if i == 0: # if its the first layer...
                layer_input_dim=self._input_dim
            else: # Must not be the first layer.
                layer_input_dim=self._layers[-1].output_dim

            self.add_layer(
                FullyConnectedLayer(
                    input_dim=layer_input_dim,
                    output_dim=l[0],
                    weight_initializer=weight_initializer,
                    activation=l[1]
                ))

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
        self._forward_cache = {}
        self._parameters = {}
        self._gradients = {}

        # Initialize model parameters.
        self._parameters['W'] = weight_initializer.initialize(np.zeros((input_dim, output_dim)))
        self._parameters['b'] = np.zeros((1, output_dim))
        
        if(initialize_bias):
            self._parameters['b'] = weight_initializer.initialize(self._parameters['b'])

        if self._activation == 'linear':
            self._activation_fx = linear
        elif self._activation == 'sigmoid':
            self._activation_fx = sigmoid
        else:
            raise ValueError('Activation function not implemented.')

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Get model parameters.
        W = self._parameters['W']
        b = self._parameters['b']

        # make sure dimmenions match.
        assert X.shape[1] == W.shape[0]
        #print(X.shape, self._weights.shape, self._bias.shape) 
        Z = np.matmul(X, W) + b 
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

    def backward(self, dL: np.ndarray) -> np.ndarray:
        A_prev = self._forward_cache['A_prev']
        Z = self._forward_cache['Z']
        A = self._forward_cache['A']
        dz = dL * self._get_activation_derivative(A)
        dw = (np.matmul(A_prev.T, dz))/A_prev.shape[0]
        db = np.mean(dz, axis=0, keepdims=True)

        # Cache the gradients for use by the optimizer.
        self._gradients['W'] = dw
        self._gradients['b'] = db
 
        loss_output = np.matmul(dz, self._parameters['W'].T)
        return loss_output

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def gradients(self) -> Dict[str, Any]:
        return self._gradients

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

