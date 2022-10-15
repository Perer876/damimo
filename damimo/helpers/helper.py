import pandas as pd


def frecuency(data_set, attribute_name, class_name=None):
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


