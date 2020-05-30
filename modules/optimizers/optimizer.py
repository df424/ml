
from abc import ABC, abstractmethod
from ml.models import Model

class Optimizer(ABC):
    @abstractmethod
    def update(self, model: Model):
        pass