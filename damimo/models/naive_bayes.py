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
            sums = frecuency_table.sum(axis=0)

            # Generamos tabla de verosimilitud
            likelihood_table = frecuency_table.apply(lambda x: x.apply(lambda y: Fraction(y, sums[x.name])))
            self.likelihood_tables[attribute] = likelihood_table

        # Generamos tabla de frecuencia para la clase y su tabla de verosimilitud
        frecuency_table = helper.frecuency(data_set, self.class_col)
        sum_ = frecuency_table.sum()
        self.likelihood_tables[self.class_col] = frecuency_table.apply(lambda x: Fraction(x, sum_))

    def predict(self, instance: pd.Series):
        pass


file_path = "C:\\Users\\peter\\Documents\\Escuela\\7mo Semestre\\Minería de Datos\\damimo\\lenses.csv"
columns = ["age", "prescription", "astigmatic", "tear_rate", "lenses"]
class_ = "lenses"
attribute_types = []

df = pd.read_csv(
    file_path,
    skiprows=[0, 1, 2],
    names=columns,
)

nb = NaiveBayes(class_)
nb.train(df)
# print("------------------------------")
# for col, ft in nb.likelihood_tables.items():
#     print(ft, "\n\n")
