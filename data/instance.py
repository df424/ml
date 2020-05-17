
from typing import Dict, Any

class Instance:
    def __init__(self, x: Any, y: Any, **kwargs):
        self.x = x
        self.y = y
        self.meta = kwargs