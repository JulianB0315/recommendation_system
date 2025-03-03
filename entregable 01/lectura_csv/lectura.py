import numpy as np
import pandas as pd

# Función para cargar datos con Pandas
def cargar_datos_pandas(ruta_archivo):
    return pd.read_csv(ruta_archivo)

# Función para cargar datos con Numpy
def cargar_datos_numpy(ruta_archivo):
    return np.genfromtxt(ruta_archivo, delimiter=',', skip_header=1)

# Ejemplo de uso
if __name__ == "__main__":
    datos_pandas = cargar_datos_pandas('ventas.csv')
    print(datos_pandas.head())

    datos_numpy = cargar_datos_numpy('ventas.csv')
    print(datos_numpy[:5])