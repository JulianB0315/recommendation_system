from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Cargar datos
datos = pd.read_csv('../data/ventas.csv')

# Preprocesamiento de datos
X = datos.drop('producto_comprado', axis=1)
y = datos['producto_comprado']

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Predecir y evaluar
predicciones = modelo.predict(X_test)
precision = accuracy_score(y_test, predicciones)
print(f'Precisión del modelo: {precision}')