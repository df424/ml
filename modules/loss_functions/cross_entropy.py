

import numpy as np
import sys

def cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> float:
    # pylint: disable=assignment-from-no-return
    norm_y_hat = np.maximum(y_hat, sys.float_info.epsilon)
    loss = -(y*np.log(norm_y_hat)).sum()
    return loss

def cross_entropy_update(X: np.ndarray, Y: np.ndarray, Y_hat: np.ndarray):
    grads = np.matmul(X.T, Y_hat-Y)/X.shape[0]
    return grads
