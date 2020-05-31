
from ml.modules.loss_functions import LossFunction
import numpy as np

class SquaredError(LossFunction):
    def scaler_loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        err = y-y_hat
        return (np.dot(err.T, err)/y.shape[0]).squeeze()

    def gradient(self, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        return -np.sum(y-y_hat)/y.shape[0]


