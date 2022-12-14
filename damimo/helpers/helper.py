import pandas as pd
from math import pi, e, sqrt, dist
from sys import float_info

__sqrt_2pi = sqrt(2 * pi)  # La raiz cuadra de 2 multiplicado por pi
__small_number = sqrt(float_info.min)


def frecuency(data_set: pd.DataFrame, attribute_name, class_name=None):
    """
    Devuelve una tabla de frecuencia simple si solo se proporciona el el nombre
    del atributo, es decir, cuenta la apacición de cada valor por esa columna.

    Si se suministra también un valor de clase, se devuelve una tabla de frecuencia
    completa, es decir, cuanta la frecencia de cada valor de la columna por cada
    valor de la columna de la clase especificada.
    """
    if class_name is None:
        return data_set[attribute_name].value_counts()
    else:
        return pd.crosstab(index=data_set[attribute_name], columns=data_set[class_name])


def split(data_set: pd.DataFrame, train_frac: float = 1):
    """
    Divide un conjunto de datos en dos, uno para entrenamiento
    y otra para pruebas. El tamaño depende del porcentaje especificado.
    """
    train_data_set = data_set.sample(frac=train_frac)
    test_data_set = data_set.drop(train_data_set.index)
    return train_data_set, test_data_set


def values_per_class(data_set: pd.DataFrame, attribute_name, class_name=None):
    """
    Devuelve los valores del atributo especificado que coincidan con la
    clase.
    """
    values = {}
    for class_value in data_set[class_name].unique():
        values[class_value] = data_set[data_set[class_name] == class_value][attribute_name]
    return values


def pdf(x, mean, std):
    """
    Probability Density Function (PDF).
    """
    # Evitamos dividir por 0 asignandole un valor muy pequeño.
    if std == 0:
        std = __small_number
    factor1 = 1 / (__sqrt_2pi * std)
    factor2 = e ** -(((x - mean) ** 2) / (2 * std ** 2))
    return factor1 * factor2


def get_pdf_for(values: pd.Series):
    """
    Devuelve la función de densidad de probabilidad asociada
    a un arreglo de valores.
    """
    mean = values.mean()
    std = values.std(ddof=0)

    def short_pdf(x):
        return pdf(x, mean, std)

    return short_pdf


def compare(values: pd.Series, other_values: pd.Series):
    """
    Compara dos arreglos de datos y devuelve los aciertos
    (cantidad de veces que se emparejaron los valores) contra
    la cantidad de valores probados.
    """
    success: pd.Series = values == other_values
    success_count = success.value_counts()
    return success_count[True] if True in success_count else 0, success.count()


def euclidean_distance(values: pd.Series, other_values: pd.Series):
    """
    Calcula la distancia euclidiana entre dos puntos dados.
    """
    return dist(values, other_values)


def manhattan_distance(values: pd.Series, other_values: pd.Series):
    """
    Calcula la distancia de manhattan entre dos puntos dados.
    """
    result = 0.0
    for col in values.index:
        result += abs(values[col] - other_values[col])
    return result


def hamming_distance(values: pd.Series, other_values: pd.Series):
    """
    Calcula la distancia de hamming entre dos puntos dados.
    """
    result = 0
    for col in values.index:
        if values[col] != other_values[col]:
            result += 1
    return result


def mixed_dist(values: pd.Series, other_values: pd.Series, discrete_attr=None):
    """
    Calcula la distancia entre dos vectores con atributos mixtos (categoricos
    y númericos)
    """
    if discrete_attr is None:
        discrete_attr = []
    result = 0.0
    for col in values.index:
        if col in discrete_attr:
            if values[col] != other_values[col]:
                result += 1
        else:
            result += abs(values[col] - other_values[col])
    return result


def get_mixed_dist_func_for(discrete_attr):
    """
    Devuelve una función de distancia mixta que trabaja que ya
    conoce cuales son los atributos discretos.
    """
    def mixed_dist_func(values: pd.Series, other_values: pd.Series):
        return mixed_dist(values, other_values, discrete_attr)

    return mixed_dist_func
