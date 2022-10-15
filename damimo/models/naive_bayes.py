"""
Naïve Bayes Model
"""
import pandas as pd
from damimo import Model
from damimo.helpers import helper


class NaiveBayes(Model):
    def __init__(self, class_col):
        self.class_col = class_col

    def train(self, data_set: pd.DataFrame):
        pass

    def predict(self, instance: pd.Series):
        pass
