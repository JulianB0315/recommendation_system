from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
datos = pd.read_csv('data/ventas.csv')

# Preprocesamiento de datos
X = datos.drop('producto_comprado', axis=1)
y = datos['producto_comprado']

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(X_train, y_train)

# Predecir productos que podrían interesar a los usuarios
predicciones = modelo.predict(X_test)

# Imprimir predicciones
print("Predicciones:")
print(predicciones)

# Matriz de confusión
matriz_confusion = confusion_matrix(y_test, predicciones)
print("\nMatriz de Confusión:")
print(matriz_confusion)

# Reporte de clasificación
reporte_clasificacion = classification_report(y_test, predicciones)
print("\nReporte de Clasificación:")
print(reporte_clasificacion)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()