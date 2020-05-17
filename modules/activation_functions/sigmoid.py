
import numpy as np

def sigmoid(Y: np.ndarray):
    return 1/(1+np.exp(-Y))
