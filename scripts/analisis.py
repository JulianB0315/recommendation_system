import pandas as pd

# Función para cargar datos desde un archivo CSV
def cargar_datos(ruta_archivo):
    return pd.read_csv(ruta_archivo)

# Función para analizar datos de ventas
def analizar_ventas(datos):
    resumen = datos.describe()
    ventas_por_producto = datos.groupby('producto_id')['cantidad'].sum()
    return resumen, ventas_por_producto

# Nueva función para analizar el comportamiento del usuario
def analizar_comportamiento_usuario(datos):
    compras_por_usuario = datos.groupby('usuario_id')['producto_comprado'].sum()
    return compras_por_usuario

# Ejemplo de uso
if __name__ == "__main__":
    datos_ventas = cargar_datos('data/ventas.csv')
    resumen, ventas_por_producto = analizar_ventas(datos_ventas)
    print(resumen)
    print(ventas_por_producto)
    
    comportamiento_usuario = analizar_comportamiento_usuario(datos_ventas)
    print(comportamiento_usuario)