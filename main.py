import pandas as pd
from damimo import ZeroR, OneR, NaiveBayes, Tester
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
    main()
