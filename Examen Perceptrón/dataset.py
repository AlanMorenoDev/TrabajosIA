import pandas as pd
import numpy as np

# Generar datos aleatorios
np.random.seed(42)
n_samples = 200  # Número de muestras

# Precio relativo (x1): valores entre 0 y 1, redondeados a un decimal
x1 = np.round(np.random.rand(n_samples), 1)

# Calidad percibida (x2): valores entre 0 y 1, redondeados a un decimal
x2 = np.round(np.random.rand(n_samples), 1)

# Etiquetas (y): 1 si (x1 <= 0.6) y (x2 >= 0.7), de lo contrario 0
y = np.where((x1 <= 0.6) & (x2 >= 0.7), 1, 0)

# Crear un DataFrame
dataset = pd.DataFrame({
    'Precio relativo': x1,
    'Calidad percibida': x2,
    'Aceptación': y
})

# Guardar en un archivo CSV
file_path = 'datosEntrenamiento.csv'
dataset.to_csv(file_path, index=False)

print(f"Dataset generado y guardado en {file_path}")
