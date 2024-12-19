import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Cargar los datos desde el archivo data.pickle
def load_data():
    with open('data/data.pickle', 'rb') as pik:
        data = pickle.load(pik)

    # Barajar los datos
    np.random.shuffle(data)

    features = []
    labels = []

    # Separar las imágenes y las etiquetas
    for img, label in data:
        features.append(img)
        labels.append(label)

    # Convertir a arrays de NumPy y reordenar las dimensiones
    features = np.array([np.transpose(img, (1, 0, 2)) for img in features], dtype=np.float32) / 255.0
    labels = np.array(labels)

    return features, labels

# Cargar los datos
features, labels = load_data()

# Confirmar las dimensiones
print(f"Dimensiones de las imágenes: {features.shape}")  # Debe ser (n_samples, 28, 31, 3)

# Dividir los datos en conjunto de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=42)

# Categorías de imágenes
categories = ['bugatti_chiron', 'chevrolet_camaro', 'chevy_pop', 'lamborghini_aventador', 'mini_cooper']

# Cargar el modelo previamente entrenado
model_path = 'modelo/mymodel.keras'
model = tf.keras.models.load_model(model_path)
print(f"Modelo cargado exitosamente desde {model_path}")

# Evaluar el modelo en el conjunto de prueba
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss}")
print(f"Precisión en el conjunto de prueba: {accuracy}")

# Hacer predicciones en el conjunto de prueba
predictions = model.predict(x_test)

# Listas para almacenar las imágenes y predicciones incorrectas/correctas
incorrect_predictions = []
correct_predictions = []

for i in range(len(x_test)):
    actual_label = categories[y_test[i]]
    predicted_label = categories[np.argmax(predictions[i])]

    # Guardar imágenes incorrectas o correctas
    if actual_label != predicted_label:
        incorrect_predictions.append((x_test[i], actual_label, predicted_label))
    else:
        correct_predictions.append((x_test[i], actual_label, predicted_label))

print(f"Total imágenes incorrectas: {len(incorrect_predictions)}")
print(f"Total imágenes correctas: {len(correct_predictions)}")

# Limitar el número de imágenes a mostrar
max_images = 15

# Función para mostrar las imágenes en galería
def show_images_in_gallery(images, titles, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.ravel()

    for i in range(min(len(images), rows * cols)):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i], fontsize=8)  # Títulos más pequeños
        axes[i].axis('off')

    # Ocultar subgráficos vacíos
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Mostrar imágenes incorrectas
if incorrect_predictions:
    print("\nMostrando imágenes incorrectas (máximo 15):")
    images = [img for img, _, _ in incorrect_predictions[:max_images]]
    titles = [f"Real: {actual}\nPred: {predicted}" for _, actual, predicted in incorrect_predictions[:max_images]]
    show_images_in_gallery(images, titles, 3, 5)

# Mostrar imágenes correctas
if correct_predictions:
    print("\nMostrando imágenes correctas (máximo 15):")
    images = [img for img, _, _ in correct_predictions[:max_images]]
    titles = [f"Real: {actual}\nPred: {predicted}" for _, actual, predicted in correct_predictions[:max_images]]
    show_images_in_gallery(images, titles, 3, 5)

# Generar el reporte de clasificación
report = classification_report(y_test, np.argmax(predictions, axis=1), target_names=categories, zero_division=1)

# Imprimir el reporte de clasificación
print("\n=== Reporte de Clasificación ===")
print(report)

# Cargar el historial previamente guardado si lo tienes
history_path = 'history.pkl'
if os.path.exists(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(accuracy))

    # Graficar precisión
    plt.plot(epochs, accuracy, 'bo', label='Precisión de entrenamiento')
    plt.plot(epochs, val_accuracy, 'b', label='Precisión de validación')
    plt.title('Precisión durante el entrenamiento')
    plt.legend()

    # Graficar pérdida
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Pérdida de entrenamiento')
    plt.plot(epochs, val_loss, 'b', label='Pérdida de validación')
    plt.title('Pérdida durante el entrenamiento')
    plt.legend()

    plt.show()
else:
    print(f"No se encontró el archivo de historial en '{history_path}'.")
