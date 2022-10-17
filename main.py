import pandas as pd
from damimo import ZeroR, Tester


def main():
    # Cargar los datos
    df = pd.read_csv(
        "db.csv",
        skiprows=[0],
        names=["Compra", "Mantenimiento", "Puertas", "Personas", "Clase"],
    )

    # Creamos el modelo (para one-r y naive bayes puede haber mas parametros)
    zr = ZeroR("Clase")

    # Instanciamos el tester.
    #   Le pasamos el modelo
    #   Y el conjunto de datos completo
    tester = Tester(zr, df)

    # Ejecutamos las pruebas.
    #   Indicamos que tome el 70% para conjunto de entrenamiento.
    #   Y que repita el entrenamiento 50 veces (en cero-r solo se necesita 1)
    result = tester.test(0.7, 10000)

    # Mostramos resultados (o lo vaciamos en un archivo)
    print(result)


if __name__ == "__main__":
    main()
