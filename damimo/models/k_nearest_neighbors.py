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
        # Calculamos las distancias entre la instancia
        distances = self.distances(instance)

        # Obtenemos las distancias mas pequeñas
        smallest_distances = self.smallest_values(distances)

        # Obtenemos las clases de las distancias mas pequeñas
        classes = self.data_set.iloc[smallest_distances.index][self.class_col]

        # Predecimos el valor con esas clasess
        return self.calculate(classes)

    def distances(self, instance: pd.Series) -> pd.Series:
        """
        Debe retornar un Series con las distancias entre la
        instancia y el conjunto de datos. Cada distancia debe
        recordar el indice de su instancia en el conjunto de datos
        """
        result = pd.Series(dtype=float, index=self.data_set.index)
        for index, other_instance in self.data_set.loc[:, self.data_set.columns != self.class_col].iterrows():
            result[index] = self.dist(instance, other_instance)
        return result

    def smallest_values(self, values: pd.Series) -> pd.Series:
        """
        Debe regresar los k valores mas pequeños en un arreglo.
        Se basa en el k de la clase
        """
        return values.nsmallest(self.k)

    def calculate(self, nearest_class_values: pd.Series):
        """
        Debe recibir el valor de las clases mas cercanas y devuelve la
        predicción para las mismas
        """
        if self.regression:
            return nearest_class_values.mean()

        mode = nearest_class_values.mode()
        return mode[0]
