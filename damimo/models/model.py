"""
Abstract Model
"""
from abc import ABC, abstractmethod


class Model(ABC):
    class_col: str

    @abstractmethod
    def train(self, data_set):
        pass

    @abstractmethod
    def predict(self, row):
        pass

    @abstractmethod
    def predict_all(self, data_set):
        pass
