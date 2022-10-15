from damimo import Model
from dataclasses import dataclass


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

    def train(self, data_set):
        pass

    def predict(self, row):
        pass

    def predict_all(self, data_set):
        pass


