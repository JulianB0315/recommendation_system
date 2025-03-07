import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm  

# Definir una red neuronal simple
class RedNeuronal(nn.Module):
    def __init__(self):
        super(RedNeuronal, self).__init__()
        self.fc1 = nn.Linear(4, 50)  
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Cargar datos
datos = pd.read_csv('data/ventas.csv')

# Preprocesamiento de datos
X = datos.drop('producto_comprado', axis=1).values
y = datos['producto_comprado'].values

# Convertir a tensores de PyTorch
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Crear una instancia de la red
modelo = RedNeuronal()

# Definir una función de pérdida y un optimizador
criterio = nn.BCELoss()
optimizador = optim.SGD(modelo.parameters(), lr=0.01)

# Ejemplo de entrenamiento con barra de progreso
num_epochs = 100
for epoch in tqdm(range(num_epochs), desc="Entrenamiento", unit="época"):
    optimizador.zero_grad()
    salida = modelo(X_train)
    perdida = criterio(salida, y_train)
    perdida.backward()
    optimizador.step()
    
    # Mostrar la pérdida en cada época
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Época [{epoch + 1}/{num_epochs}], Pérdida: {perdida.item():.4f}")

print('Entrenamiento completado')