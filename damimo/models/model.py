"""
Abstract Model
"""
from abc import ABC, abstractmethod


class Model(ABC):
    class_col: str

    @abstractmethod
    def train(self, data_set):
        """
        El metodo que se encargará de entrenar al modelo con el
        conjunto de datos que se suministre.
        """
        pass

    @abstractmethod
    def predict(self, instance):
        """
        Se encargara de predecir el valor de clase para una sola
        isntancia del conjunto.
        """
        pass

    @abstractmethod
    def predict_all(self, data_set):
        """
        Se encargara de predecir todos los valores de clase para
        todas las instancias del conjunto de datos que se suministre.
        """
        pass
