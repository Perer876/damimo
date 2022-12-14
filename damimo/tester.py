"""
Model Tester
"""
import pandas as pd
from damimo import Model
from damimo.helpers import helper


class Tester:
    def __init__(self, model: Model, data_set: pd.DataFrame):
        self.model = model
        self.data_set = data_set

    def split(self, train_frac: float):
        """
        Divide  el conjunto de datos interno en dos,uno para
        entrenamiento y otro para prueba.
        """
        return helper.split(self.data_set, train_frac)

    def test_once(self, train_frac: float):
        """
        Hace la prueba para el conjunto de datos interno solo una
        vez y devuelve el resultado (aciertos / total de instancias)
        según el % para entrenaiento.

        Si train_frac es 1, entrena y prueba con el conjunto entero.
        """
        # Obtenemos los conjuntos de entrenamiento y prueba.
        if train_frac == 1:
            train_df, test_df = self.data_set, self.data_set
        else:
            train_df, test_df = self.split(train_frac)

        # Entrenamos el modelo
        self.model.train(train_df)

        # Vemos los valores que asigna el modelo a las isntancias de prueba
        predict = self.model.predict_all(test_df)

        # Comparar predict con las clases correctas y determinar aciertos y
        # cantidad de instancias
        return helper.compare(predict, test_df[self.model.class_col])

    def test(self, train_frac: float, times=1):
        """
        Prueba el conjunto de datos la cantidad de veces indicada y
        el % de instancias para entrenamiento. Se apoya en el metodo
        test_once
        """
        success = 0
        instances = 0
        for i in range(0, times):
            ind_succ, ind_inst = self.test_once(train_frac)
            success += ind_succ
            instances += ind_inst
        mean = success / instances
        return success, instances, mean
