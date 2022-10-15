import pandas as pd
from damimo import ZeroR, frecuency


def main():
    df = pd.read_csv(
        "db.csv",
        skiprows=[0],
        names=["Compra", "Mantenimiento", "Puertas", "Personas", "Clase"],
    )
    print(frecuency(df, "Mantenimiento", "Clase"))


if __name__ == "__main__":
    main()
