
from typing import Any, List
import numpy as np
from PIL import Image
import sys

from ml.data import Instance
from ml.data.vectorizers.vectorizer import Vectorizer
from ml.data.readers import DatasetReader
from ml.utility import label_map_2_one_hot, add_bias
from tqdm.autonotebook import tqdm

class YaleFaceResampleVectorizer(Vectorizer):
    def __init__(self, add_bias=True, flatten: bool = True, mode='concat'):
        self._label_map = {}
        self._add_bias = add_bias
        self._mode = mode
        self._flatten = flatten

    def index_vectorizer(self, reader: DatasetReader):
        X = []
        for inst in tqdm(reader.read(), desc='Indexing YaleFaceResampleVectorizer'):
            # if we have not seen this label before add it to the label map and just
            # assign it the next integer since its a 1-hot thing.
            if inst.y not in self._label_map:
                self._label_map[inst.y] = len(self._label_map)
            X.append(YaleFaceResampleVectorizer._flatten_and_resize_img(inst.x, self._flatten))

        if self._mode == 'concat':
            X = np.concatenate(X)
        elif self._mode == 'array':
            X = np.array(X)
        else:
            raise ValueError(f'MODE={self._mode} not supported.')

        self._means = X.mean(axis=0)
        self._std = X.std(axis=0)

        # Handle standard deviations of 0 by making them one
        # and saving a mask so we can set all those features to zero.
        self._feature_mask = np.argwhere(self._std < sys.float_info.epsilon)
        # This avoids divide by 0 and doesn't matter since we aren't going to use
        # the feature anyway.
        self._std[self._feature_mask] = 1

        self._output_2_label = [None] * len(self._label_map)
        for k,v in self._label_map.items():
            self._output_2_label[v] = k

    def vectorize_input(self, input: Any) -> np.ndarray:
        x = YaleFaceResampleVectorizer._flatten_and_resize_img(input, self._flatten)
        x_norm = (x-self._means)/self._std
        x_norm[self._feature_mask] = 0
        if self._add_bias:
            return add_bias(x_norm)
        else:
            return x_norm

    def vectorize_label(self, label: Any) -> np.ndarray:
        return label_map_2_one_hot(label, self._label_map)

    @property
    def output_size(self) -> int:
        return len(self._label_map)

    @property
    def input_size(self) -> int:
        if self._add_bias:
            return 1601
        return 1600

    @staticmethod
    def _flatten_and_resize_img(img, flatten: bool):
        if flatten:
            return np.asarray(img.resize((40,40), resample=Image.LANCZOS)).reshape(1,-1)
        else:
            return np.asarray(img.resize((40,40), resample=Image.LANCZOS))

