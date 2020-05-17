
from typing import Any, Dict
import numpy as np

def label_map_2_one_hot(label: Any, label_map: Dict[Any, int]):
    rv = np.zeros((1, len(label_map)))
    rv[:, label_map[label]] = 1
    return rv
