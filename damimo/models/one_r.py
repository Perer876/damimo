"""
One-R Model
"""
import pandas as pd
from dataclasses import dataclass
from damimo import Model
from damimo.helpers import helper


@dataclass
class Rule:
    answer: object
    errores: tuple | list


class Rules:
    pass


class OneR(Model):
    frecuency_tables: dict
    attributes: list

    def __init__(self, class_col, attributes):
        self.class_col = class_col
        self.attributes = attributes

    def train(self, data_set: pd.DataFrame):
        pass

    def predict(self, instance: pd.Series):
        pass
