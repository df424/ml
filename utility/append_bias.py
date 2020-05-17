
import numpy as np

def add_bias(X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((X.shape[0],1)), X], axis=1)