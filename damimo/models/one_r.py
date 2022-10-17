"""
One-R Model
"""

import pandas as pd
from dataclasses import dataclass
from damimo import Model
from damimo.helpers import helper

@dataclass
class Rule:
    def __init__(self, attribute, attr_value, class_value, errors):
        self.attribute = attribute
        self.attr_value = attr_value
        self.class_value = class_value
        self.errors:tuple = errors
    
    def __str__(self) -> str:
        result = str(self.attribute)
        result+=" "
        result+= str(self.attr_value)
        result+=" "
        result+=str(self.class_value)
        result+=" "
        for item in self.errors:
            result+=str(item)
            result+=" "
        return result
    
class Rules:
    rules: dict = {}


class OneR(Model):
    frecuency_tables: dict = {}
    attributes: list
    rules: dict = {}
    model: tuple = ()

    def __init__(self, class_col):
        self.class_col = class_col

    def train(self, data_set: pd.DataFrame):
        self.attributes = self.get_attributes(data_set)

        # Generamos las tablas de frecuencia para los atributos 
        for attribute in self.attributes:
            # Declarar una lista que contendrá las reglas por atributo
            list_rules = []
            # Generamos tabla de frecuencia
            frecuency_table = helper.frecuency(data_set, attribute, self.class_col)
            # Lista de la nueva columna a la tabla de frecuencia
            total_column = []
            # Acceder al valor de frecuencia del atributo valor
            for index, instance in frecuency_table.iterrows():
                # Acumulador suma de frecuencia de valores de clase por instancia 
                total = 0
                for col in frecuency_table.columns:
                    if col != attribute:
                        #Extraer valor de frecuencia                         
                        total+=instance[str(col)]
                
                # Cada elemento de la lista será la suma de las frecuencias por valor de atributo
                total_column.append(total)
                # Crear reglas con el valor de clase más frecuente para cada renglon de la tabla de frecuencia
                att_value = index
                class_value = instance.idxmax()
                errors = (total - instance.max(), total)
                rule = Rule(attribute, att_value, class_value, errors)
                
                list_rules.append(rule)
            
            # Insertamos la columna de suma de frecuencias a la tabla
            frecuency_table['Total'] = total_column
            # Guardamos cada tabla de frecuencia por atributo en un diccionario
            self.frecuency_tables[attribute] = frecuency_table
            # Calcular error total por atributo
            num = 0
            den = 0
            for item in list_rules:
                num+= item.errors[0]
                den+= item.errors[1]
            total_error = (num, den)
            # Agrega a la lista de reglas cada conjunto de reglas por atributo y el error total
            self.rules[attribute] = (list_rules, total_error)
        # Teniendo del conjunto de reglas, generamos el modelo
        self.generate_model()    
    
    # Encuentra el conjunto de reglas con el menor error total
    def generate_model(self,):
        first_k = list(self.rules.keys())[0]
        min_total_error = self.rules[first_k][1][0]
        self.model = (first_k, self.rules[first_k])
        
        for key in self.rules:
            min_cur_error = self.rules[key][1][0]
            if min_cur_error < min_total_error:
                min_total_error = min_cur_error
                self.model = (key, self.rules[key])
            
    def predict(self, instance: pd.Series):
        # Recorre las reglas del modelo 
        for rule in self.model[1][0]:
            # Al coincidir los valores del atributo del modelo, retorna el valor de clase para dicha regla
            if rule.attr_value == (instance[self.model[0]]):
                return rule.class_value