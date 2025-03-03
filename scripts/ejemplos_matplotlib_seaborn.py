import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cargar datos
datos = pd.read_csv('../data/ventas.csv')

# Histograma con Matplotlib
plt.hist(datos['cantidad'], bins=10)
plt.xlabel('Cantidad')
plt.ylabel('Frecuencia')
plt.title('Histograma de Cantidad')
plt.show()

# Gráfico de dispersión con Seaborn
sns.scatterplot(x='cantidad', y='precio', data=datos)
plt.xlabel('Cantidad')
plt.ylabel('Precio')
plt.title('Gráfico de Dispersión de Cantidad vs Precio')
plt.show()