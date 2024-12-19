import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar los datos recolectados
datos = pd.read_csv('datosEntrenamiento.csv')

# Separar las características (X) y la etiqueta (y)
X = datos[['Velocidad', 'Distancia']].values
y = datos['Salto'].values

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal multicapa
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),  # Capa oculta con 8 neuronas y ReLU
    Dense(4, activation='relu'),              # Capa oculta adicional con 4 neuronas y ReLU
    Dense(1, activation='sigmoid')            # Capa de salida con activación sigmoide
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión en el conjunto de prueba: {accuracy:.2f}")

# Guardar el modelo entrenado
model.save('modelo_mlp.h5')
print("Modelo guardado como 'modelo_mlp.h5'.")
