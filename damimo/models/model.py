"""
Abstract Model
"""
from abc import ABC, abstractmethod
import pandas as pd


class Model(ABC):
    class_col: str
    attributes: tuple

    @abstractmethod
    def train(self, data_set: pd.DataFrame):
        """
        El metodo que se encargará de entrenar al modelo con el
        conjunto de datos que se suministre.
        """
        pass

    @abstractmethod
    def predict(self, instance: pd.Series):
        """
        Se encargara de predecir el valor de clase para una sola
        instancia del conjunto. Si la instancia contiene la clase,
        simplemente no tomarla en cuenta.

        Devuelve solo el valor y esta mas que nada hecha para uso interno.
        """
        pass

    def predict_all(self, data_set: pd.DataFrame):
        """
        Se encargara de predecir todos los valores de clase para
        todas las instancias del conjunto de datos que se suministre.
        Debe devolver las clases ya predecidas.

        Devuelve un objeto de tipo pd.Series
        """
        predictions = []

        for instance in data_set.iterrows():
            predictions.append(self.predict(instance[1]))

        return pd.Series(
            data=predictions,
            index=data_set.index
        )

    def get_attributes(self, data_set: pd.DataFrame):
        """
        Devuelve una lista con los atributos de determinado conjunto
        de datos. Lo que hace es tomar la lista de columnas y excluye
        la columna clase.
        """
        return tuple(col for col in data_set.columns if col != self.class_col)
