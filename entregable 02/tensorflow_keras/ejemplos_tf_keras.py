import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

# Cargar datos
datos = pd.read_csv('../../ventas.csv')

# Preprocesamiento de datos
X = datos.drop('producto_comprado', axis=1).values
y = datos['producto_comprado'].values

# Definir un modelo secuencial
modelo = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
modelo.fit(X, y, epochs=10, batch_size=32)

print('Entrenamiento completado')