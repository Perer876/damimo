"""
K-Nearest Neighbors
"""
import pandas as pd
from damimo import Model
from damimo.helpers import helper


class KNearestNeighbors(Model):
    data_set: pd.DataFrame

    def __init__(self, class_col, k: int = 1, regression=False, dist=helper.euclidean_distance):
        self.class_col = class_col
        self.k = k
        self.regression = regression
        self.dist = dist

    def train(self, data_set: pd.DataFrame):
        self.data_set = data_set

    def predict(self, instance: pd.Series):
        distances = self.distances(instance)
        smallest_distances = self.smallest_values(distances)

        # De los indices de las distancias mas cercanas, sacar las
        # clases mas cercanas
        nearest_classes = pd.Series()

        return self.calculate(nearest_classes)

    def distances(self, instance: pd.Series) -> pd.Series:
        """
        Debe retornar un Series con las distancias entre la
        instancia y el conjunto de datos. Cada distancia debe
        recordar el indice de su instancia en el conjunto de datos
        """
        pass

    def smallest_values(self, values: pd.Series) -> pd.Series:
        """
        Debe regresar los k valores mas pequeños en un arreglo.
        Se basa en el k de la clase
        """
        pass

    def calculate(self, nearest_class_values: pd.Series):
        """
        Debe recibir el valor de las clases mas cercanas y devuelve la
        predicción para las mismas
        """
        pass

