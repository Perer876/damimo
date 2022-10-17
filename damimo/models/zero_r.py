"""
Zero-R Model
"""
import pandas as pd
from damimo import Model
from damimo.helpers import helper


class ZeroR(Model):
    frecuency_table: pd.Series
    most_frecuent_class: object

    def __init__(self, class_col):
        self.class_col = class_col

    def train(self, data_set: pd.DataFrame):
        self.frecuency_table = helper.frecuency(data_set, self.class_col)
        self.most_frecuent_class = self.frecuency_table.idxmax()

    def predict(self, instance: pd.Series = None):
        return self.most_frecuent_class
