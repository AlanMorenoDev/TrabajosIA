import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el dataset desde un archivo CSV
file_path = 'datosEntrenamiento.csv'
dataset = pd.read_csv(file_path)

# Asegurarnos de que solo tomamos las columnas necesarias
# Asumimos que las dos primeras columnas son características (x1, x2) y la tercera es la etiqueta (y)
X = dataset.iloc[:, :2].values  # Características (x1, x2)
y = dataset.iloc[:, 2].values   # Etiquetas (0 o 1)

# Hiperparámetros del perceptrón
learning_rate = 0.1
epochs = 100

# Inicialización de pesos y sesgo
np.random.seed(42)
weights = np.random.rand(2)
bias = np.random.rand()

# Función de activación escalón
def step_function(z):
    return 1 if z >= 0 else 0

# Entrenamiento del perceptrón
for epoch in range(epochs):
    for i in range(len(X)):
        # Predicción
        z = np.dot(weights, X[i]) + bias
        y_pred = step_function(z)
        
        # Actualización de pesos y sesgo
        error = y[i] - y_pred
        weights += learning_rate * error * X[i]
        bias += learning_rate * error

# Resultados finales
print("Pesos finales:", weights)
print("Sesgo final:", bias)

# Graficar datos y frontera de decisión
plt.figure(figsize=(8, 6))

# Puntos de datos
for i in range(len(y)):
    if y[i] == 1:
        plt.scatter(X[i, 0], X[i, 1], color='green', label='Aceptado' if i == 0 else "", s=50)
    else:
        plt.scatter(X[i, 0], X[i, 1], color='red', label='Rechazado' if i == 0 else "", s=50)

# Frontera de decisión: w1*x1 + w2*x2 + b = 0 => x2 = -(w1/w2)*x1 - b/w2
x1_vals = np.linspace(0, 1, 100)
x2_vals = -(weights[0] / weights[1]) * x1_vals - (bias / weights[1])
plt.plot(x1_vals, x2_vals, color='blue', label='Frontera de decisión')

# Configuración de la gráfica
plt.title('Clasificación de Productos con Perceptrón')
plt.xlabel('Precio relativo (x1)')
plt.ylabel('Calidad percibida (x2)')
plt.legend()
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
