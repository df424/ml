
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x)/np.exp(x).sum(axis=1).reshape(1,-1).T

