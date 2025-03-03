import pandas as pd
import numpy as np

# Función para leer un archivo CSV con Pandas
def leer_csv_pandas(ruta_archivo):
    return pd.read_csv(ruta_archivo)

# Función para leer un archivo CSV con Numpy
def leer_csv_numpy(ruta_archivo):
    return np.genfromtxt(ruta_archivo, delimiter=',', skip_header=1)

# Ejemplo de uso
if __name__ == "__main__":
    datos_pandas = leer_csv_pandas('../data/ventas.csv')
    print(datos_pandas.head())
    
    datos_numpy = leer_csv_numpy('../data/ventas.csv')
    print(datos_numpy)