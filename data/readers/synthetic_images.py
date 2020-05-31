
from typing import Generator, Tuple

import numpy as np
from ml.data.readers import DatasetReader
from ml.data import Instance

class SyntheticImageGenerator(DatasetReader):
    def __init__(self, dims: Tuple[int, int], strip_width: int):
        self._dims = dims
        self._strip_width = strip_width

    def read(self) -> Generator[Instance, None, None]:
        img = np.zeros(self._dims)
        start = int(self._dims[0]/2-self._strip_width/2)
        img[:, start:start+self._strip_width] = 1
        yield Instance(img, 0)

        img = np.zeros(self._dims)
        start = int(self._dims[1]/2-self._strip_width/2)
        img[start:start+self._strip_width, :] = 1
        yield Instance(img, 1)
