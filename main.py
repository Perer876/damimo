import pandas as pd
from damimo import ZeroR, Tester
from damimo.models.naive_bayes import NaiveBayes
from damimo.models.one_r import OneR

def main():
    # Cargar los datos

    # Creamos el modelo (para one-r y naive bayes puede haber mas parametros)
    #zr = ZeroR("Clase")
    
    pass
    # Instanciamos el tester.
    #   Le pasamos el modelo
    #   Y el conjunto de datos completo
    #tester = Tester(zr, df)

    # Ejecutamos las pruebas.
    #   Indicamos que tome el 70% para conjunto de entrenamiento.
    #   Y que repita el entrenamiento 50 veces (en cero-r solo se necesita 1)
    #result = tester.test(0.7, 50)

    # Mostramos resultados (o lo vaciamos en un archivo)
    #print(result)


if __name__ == "__main__":
    main()
