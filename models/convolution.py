
from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import convolve2d
import math

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

        output = []
        # For each input in the batch
        for x in X:
            channels = []
            for kernel in self._parameters['K']:
                channels.append(convolve2d(x, kernel, mode=self._mode))

            output.append(np.array(channels))
        return np.array(output)

    def backward(self, loss: np.ndarray) -> None:
        # Get our input X.
        X = self._cache['X']
        K = self._parameters['K']

        # Create a tensor to hold the output gradient.
        output_grad = np.zeros(X.shape)

        # Create a tensor to store our gradients we know it will be the same size as our filters.
        grads = np.zeros(self._parameters['K'].shape)

        # Iterate over our filters and compute the gradient for each one.
        for x, l in zip(X, loss):
            for i in range(K.shape[0]):
                grads[i] += convolve2d(x, l[i], mode=self._mode)
                #output_grad[i] += convolve2d(np.rot90(K[i], 2), l[i], mode='same')
        # take the mean of all of the batched images.
        grads = grads/X.shape[0]
        output_grad/X.shape[0]

        # Store our gradients in the cache so that the optimizer can do the updating.
        self._gradients['K'] = grads

        return output_grad

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def gradients(self) -> Dict[str, Any]:
        return self._gradients


class PoolingLayer2D(Model):
    def __init__(self, shape:Tuple[int, int], stride: Tuple[int, int], method:str='max', mode:str='valid'):
        self._shape = shape
        self._stride = stride
        self._method = method
        self._mode = mode
        self._cache = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        # We know the size of our pooling layer from the size of the images and stride so we can.
        # allocate spaces once.
        n_output_rows = math.ceil(X.shape[2]/self._stride[0])
        n_output_cols = math.ceil(X.shape[3]/self._stride[1])
        pred = np.zeros((X.shape[0], X.shape[1], n_output_rows, n_output_cols))
        # We need to keep track of the indices so we can back prop.
        indices = np.zeros(X.shape)

        # For each input image...
        #print(X.shape, pred.shape)
        for k in range(X.shape[0]):
            # For each channel...
            for n, x in enumerate(X[k]):
                if self._mode == 'valid':
                    # Iterate over rows.
                    for i in range(0, n_output_rows):
                        row_start = i*self._stride[0]
                        row_end = row_start + self._shape[0]
                        # Iterate over columns
                        for j in range(0, n_output_cols):
                            col_start = j*self._stride[1]
                            col_end = col_start + self._shape[1]
                            #print(row_start, row_end, col_start, col_end, self._shape, x.shape)
                            #print(x[row_start:row_end, col_start:col_end])
                            mi = np.unravel_index(np.argmax(x[row_start:row_end, col_start:col_end]), self._shape)
                            #print(mi[0]+row_start, mi[1]+col_start, indices.shape)
                            indices[k, n, mi[0]+row_start, mi[1]+col_start] = 1
                            #print(f'PRED_INDICES: {k}, {n}, {i}, {j}')
                            pred[k,n,i,j] = np.max(x[row_start:row_end, col_start:col_end])
                else:
                    raise ValueError(f'Unknown mode: {self._mode}')
        # cache the indices so we can use them during backprop.
        self._cache['I'] = indices
        return pred

    def backward(self, loss: np.ndarray) -> None:
        # Get the indices from teh cache.
        I = self._cache['I']
        grads = np.zeros(I.shape)

        n_output_rows = math.ceil(I.shape[2]/self._stride[0])
        n_output_cols = math.ceil(I.shape[3]/self._stride[1])

        for k in range(loss.shape[0]):
            for n in range(loss.shape[1]):
                for i in range(0, n_output_rows):
                    row_start = i*self._stride[0]
                    row_end = row_start + self._shape[0]
                    # Iterate over columns
                    for j in range(0, n_output_cols):
                        col_start = j*self._stride[1]
                        col_end = col_start + self._shape[1]
                        #print(f'GRAD INDICES: k={k}, n={n}, row_start={row_start}, row_end={row_end}, col_start={col_start}, col_end={col_end}, i={i}, j={j}')
                        grads[k, n, row_start:row_end, col_start:col_end] = loss[k,n,i,j]

        return np.multiply(grads, I)

    @property
    def parameters(self) -> Dict[str, Any]:
        return {}

    @property
    def gradients(self) -> Dict[str, Any]:
        return {}