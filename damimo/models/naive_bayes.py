"""
Naïve Bayes Model
"""
import pandas as pd
from damimo import Model
from damimo.helpers import helper
from fractions import Fraction


class NumericHandler:
    """
    Clase de ayuda para manejar los attributos continuos en el modelo
    de Naive Bayes.
    """
    def __init__(self, values: pd.Series):
        self.pdf = helper.get_pdf_for(values)

    def __getitem__(self, item):
        """
        Este se usa para poder llamar aL PDF como si se estuviera
        accediendo a un elemento de un diccionario, lo que le permite
        al algoritmo de Naibe Bayes que no le importe si esta
        accediendo a un atributo discreto o continuo al predecir.
        """
        return self.pdf(item)


class NaiveBayes(Model):
    likelihood_tables: dict = {}

    def __init__(self, class_col, numeric_attr=None):
        self.class_col = class_col
        if numeric_attr is None:
            numeric_attr = []
        self.numeric_attr = numeric_attr

    def train(self, data_set: pd.DataFrame):
        self.attributes = self.get_attributes(data_set)

        # Generamos las tablas de frecuencia para los atributos considerando el problema de frecuancia cero
        for attribute in self.attributes:
            if attribute in self.numeric_attr:
                self.likelihood_tables[attribute] = self.numeric_likelihood(data_set, attribute)
            else:
                self.likelihood_tables[attribute] = self.likelihood(data_set, attribute)

        # Generamos tabla de frecuencia para la clase y su tabla de verosimilitud
        frecuency_table = helper.frecuency(data_set, self.class_col)
        sum_ = frecuency_table.sum()
        self.likelihood_tables[self.class_col] = frecuency_table.apply(lambda x: Fraction(x, sum_))

    def likelihood(self, data_set: pd.DataFrame, attribute):
        """
        Genera la tabla de verosimilitud para un atributo discreto
        """
        # Generamos tabla de frecuencia
        frecuency_table = helper.frecuency(data_set, attribute, self.class_col)
        # Corregimos error de frecuancia cero
        frecuency_table = frecuency_table.applymap(lambda x: x + 1)
        # Calculamos la suma de ocurrencias por clase
        sums = frecuency_table.sum()
        # Generamos tabla de verosimilitud
        return frecuency_table.apply(lambda x: x.apply(lambda y: Fraction(y, sums[x.name])))

    def numeric_likelihood(self, data_set: pd.DataFrame, attribute):
        """
        Genera una diccionario que permite acceder como llave el valor de clase
        y de da el NumericHandler asociado.
        """
        values_per_class = helper.values_per_class(data_set, attribute, self.class_col)
        pdfs = {}
        for class_value, values in values_per_class.items():
            pdfs[class_value] = NumericHandler(values)
        return pdfs

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
