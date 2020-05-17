
from typing import List
import PIL
from PIL import Image
from typing import List, Generator
import os
import numpy as np
import re

from ml.data.readers import DatasetReader
from ml.data import Instance

class YaleFaceReader(DatasetReader):
    def __init__(self, files: List[str]):
        self._file_paths = files

    def read(self) -> Generator[Instance, None, None]:
        for file in self._file_paths:
            img = Image.open(file)
            file_toks = os.path.basename(file).split('.')
            label = file_toks[0]
            mode = file_toks[1]

            yield Instance(img, label, mode=mode, file_path=file)