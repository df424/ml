

import numpy as np
import sys

def log_likelihood(y: np.ndarray, y_hat: np.ndarray) -> float:
    # pylint: disable=assignment-from-no-return
    norm_y_hat = np.maximum(y_hat, sys.float_info.epsilon)
    norm_one_minus_y_hat = np.maximum(1-y_hat, sys.float_info.epsilon)
    loss = -(y * np.log(norm_y_hat) + (1-y) * np.log(norm_one_minus_y_hat)).sum(axis=1).mean()
    return loss

def log_likelihood_sigmoid_update(X: np.ndarray, Y: np.ndarray, Y_hat: np.ndarray):
    return -np.matmul(X.T, Y-Y_hat)/X.shape[0]
