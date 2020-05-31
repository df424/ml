
from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import convolve2d

from ml.models import Model
from ml.modules.initializers import Initializer

class ConvolutionalLayer2D(Model):
    def __init__(self, 
        kernel_shape:Tuple[int, int], 
        num_kernels: int, 
        mode='valid',
        initializer: Initializer = None,
    ):
        self._kernel_shape = kernel_shape
        self._num_kernels = num_kernels
        self._mode = mode
        self._cache = {}
        self._parameters = {}
        self._gradients = {}

        self._parameters['K'] = np.zeros((num_kernels, kernel_shape[0], kernel_shape[1]))

        if initializer:
            self._parameters['K'] = initializer.initialize(self._parameters['K'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._cache['X'] = X

        channels = []

        for kernel in self._parameters['K']:
            channels.append(convolve2d(X, kernel, mode=self._mode))

        return np.array(channels)

    def backward(self, loss: np.ndarray) -> None:
        # Loss should be the same size as our filters...
        assert(loss.shape == self._parameters['K'].shape)

        # Get our input X.
        X = self._cache['X']

        # Create a tensor to store our gradients we know it will be the same size as our filters.
        grads = np.zeros(loss.shape)

        # Iterate over our filters and compute the gradient for each one.
        for i, l in enumerate(loss):
            grads[i] = convolve2d(X, l, mode=self._mode)

        # Store our gradients in the cache so that the optimizer can do the updating.
        self._gradients['K'] = grads

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def gradients(self) -> Dict[str, Any]:
        return self._gradients


class PoolingLayer2D(Model):
    def __init__(self, shape:Tuple[int, int], stride: int, method:str='max', mode:str='valid'):
        self._mode = mode
        self._parameters = {}
        self._gradients = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._mode == 'valid':
            pass
        else:
            raise ValueError(f'Unknown mode: {self._mode}')

    def backward(self, loss: np.ndarray) -> None:
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def gradients(self) -> Dict[str, Any]:
        return self._gradients