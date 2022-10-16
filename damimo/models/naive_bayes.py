"""
Naïve Bayes Model
"""
import pandas as pd
from damimo import Model
from damimo.helpers import helper
from fractions import Fraction


class NaiveBayes(Model):
    likelihood_tables: dict = {}

    def __init__(self, class_col):
        self.class_col = class_col

    def train(self, data_set: pd.DataFrame):
        self.attributes = self.get_attributes(data_set)

        # Generamos las tablas de frecuencia para los atributos considerando el problema de frecuancia cero
        for attribute in self.attributes:
            # Generamos tabla de frecuencia
            frecuency_table = helper.frecuency(data_set, attribute, self.class_col)

            # Corregimos error de frecuancia cero
            frecuency_table = frecuency_table.applymap(lambda x: x + 1)

            # Calculamos la suma de ocurrencias por clase
            sums = frecuency_table.sum()

            # Generamos tabla de verosimilitud
            likelihood_table = frecuency_table.apply(lambda x: x.apply(lambda y: Fraction(y, sums[x.name])))
            self.likelihood_tables[attribute] = likelihood_table

        # Generamos tabla de frecuencia para la clase y su tabla de verosimilitud
        frecuency_table = helper.frecuency(data_set, self.class_col)
        sum_ = frecuency_table.sum()
        self.likelihood_tables[self.class_col] = frecuency_table.apply(lambda x: Fraction(x, sum_))

    def predict(self, instance: pd.Series):
        # Calcula las probabilidades por cada valor de clase
        pr = pd.Series(dtype=float)
        for class_value in self.likelihood_tables[self.class_col].index:
            pr[class_value] = self.pr(class_value, instance) * 1.0

        # Normalizamos
        pr_sum = pr.sum()
        pr = pr.apply(lambda x: x / pr_sum)

        # Devolvemos el valor maximo
        return pr.idxmax()

    def pr(self, class_value, instance: pd.Series):
        """
        Devuelve la probabilidad a posteriori para un determinado
        valor de clase e instancia.
        """
        res = 1
        for attribute in self.attributes:
            res *= self.pr_ind(attribute, instance[attribute], class_value)
        return res * self.likelihood_tables[self.class_col][class_value]

    def pr_ind(self, attribute, value, class_value):
        """
        Devuelve la probabilidad individual para un determinado
        valor de un atributo y la el valor de clase.
        """
        return self.likelihood_tables[attribute][class_value][value]
