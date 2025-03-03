from sklearn.cluster import KMeans
import pandas as pd

# Cargar datos
datos = pd.read_csv('ventas.csv')

# Datos de ejemplo
X = datos[['cantidad', 'precio']]

# Modelo de clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Predicciones
clusters = kmeans.predict(X)
print(clusters)