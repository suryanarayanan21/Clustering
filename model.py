from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class ClusteringFunction(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def run(self, tokens: list[str], embeddings: np.ndarray, distances: np.ndarray) -> pd.DataFrame:
        pass

class PreprocessFunction(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def run(self, sentences: list[str]) -> tuple[np.ndarray, np.ndarray]:
        '''
        Function that takes a list of sentences and returns a tuple of (embeddings, distance matrix)
        '''
        pass

class InputFunction(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def run(self) -> list[str]:
        pass