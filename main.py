import pandas as pd
from damimo import ZeroR, OneR, NaiveBayes, KNearestNeighbors, Tester
from damimo.helpers.helper import manhattan_distance, get_mixed_dist_func_for
import json

with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)


def main():
    columns = config["attributes"]
    columns.append(config["class"])

    # Cargar los datos
    df = pd.read_csv(
        config["data_set_path"],
        skiprows=config["skip_rows"],
        names=columns,
    )

    # Creamos el modelo
    if config["model"] == "cero-r":
        model = ZeroR(config["class"])
    elif config["model"] == "one-r":
        model = OneR(config["class"])
    else:
        if "numeric_attributes" in config:
            na = config["numeric_attributes"]
        else:
            na = []
        model = NaiveBayes(config["class"], na)

    # Instanciamos el tester.
    #   Le pasamos el modelo
    #   Y el conjunto de datos completo
    tester = Tester(model, df)

    # Ejecutamos las pruebas.
    result = tester.test(config["test"]["train_frac"], config["test"]["times"])

    # Mostramos resultados (o lo vaciamos en un archivo)
    print("Aciertos", result[0])
    print("Instancias", result[1])
    print("Promedio", result[2])


if __name__ == "__main__":
    # main()

    df = pd.DataFrame({'Foo': [10.2, 8.2, 7.1, 8.5, 9.3],
                       'Bar': [5.1, 6.2, 5.1, 5.2, 6.1],
                       'Baz': ["ho", "ha", "he", "ho", "ho"],
                       'Clase': ["A", "B", "A", "C", "C"]})

    instance = pd.Series({'Foo': 8.2,
                          'Bar': 6.2,
                          'Baz': "ha"})

    mixed_dist = get_mixed_dist_func_for(['Baz'])
    knn = KNearestNeighbors('Clase', k=2, dist=mixed_dist)
    knn.train(df)

    print(knn.predict(instance))
    # print(mixed_dist(instance, instance2))
