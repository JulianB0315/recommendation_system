from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
datos = pd.read_csv('data/ventas.csv')

# Datos de ejemplo
X = datos[['cantidad', 'precio']]

# Modelo de clustering
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Predicciones
clusters = kmeans.predict(X)

# Agregar los clusters al DataFrame
datos['cluster'] = clusters

# Visualización de los clusters
plt.figure(figsize=(10, 6))
plt.scatter(datos['cantidad'], datos['precio'], c=datos['cluster'], cmap='viridis')
plt.xlabel('Cantidad')
plt.ylabel('Precio')
plt.title('Clusters de Ventas')
plt.colorbar(label='Cluster')
plt.show()