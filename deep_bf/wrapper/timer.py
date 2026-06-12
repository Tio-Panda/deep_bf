import time
from contextlib import contextmanager
from collections import defaultdict
import torch

class Timer:
    def __init__(self):
        self.times = defaultdict(float)

    @contextmanager
    def measure(self, name):
        torch.cuda.synchronize()
        start = time.perf_counter()

        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.times[name] += time.perf_counter() - start
