
from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import convolve2d

from ml.models import Model
from ml.modules.initializers import Initializer

class ConvolutionalLayer(Model):
    def __init__(self, 
        kernel_shape:Tuple[int, int], 
        num_kernels: int, 
        mode='valid',
        initializer: Initializer = None,
    ):
        self._kernel_shape = kernel_shape
        self._num_kernels = num_kernels
        self._mode = mode
        self._parameters = {}
        self._gradients = {}

        self._parameters['K'] = np.zeros((num_kernels, kernel_shape[0], kernel_shape[1]))

        if initializer:
            self._parameters['K'] = initializer.initialize(self._parameters['K'])

    def predict(self, X: np.ndarray) -> np.ndarray:
        channels = []

        for kernel in self._parameters['K']:
            channels.append(convolve2d(X, kernel, mode=self._mode))

        return np.array(channels)

    def backward(self, loss: np.ndarray) -> None:
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def gradients(self) -> Dict[str, Any]:
        return self._gradients


class PoolingLayer(Model):
    def __init__(self):
        self._parameters = {}
        self._gradients = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def backward(self, loss: np.ndarray) -> None:
        pass

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def gradients(self) -> Dict[str, Any]:
        return self._gradients