import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Lista de categorías
categories = ['bugatti_chiron', 'chevrolet_camaro', 'chevy_popjjj', 'lamborghini_aventador', 'mini_cooper']

# Cargar el modelo previamente entrenado
model_path = 'modelo/mymodel.keras'  # Usar formato Keras actualizado si es posible
model = tf.keras.models.load_model(model_path)
print(f"Modelo cargado exitosamente desde {model_path}")

def predict_car_image(image_path):
    """
    Carga una imagen, la procesa y realiza una predicción con el modelo.
    """
    try:
        # Cargar la imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error al cargar la imagen: {image_path}")
            return

        # Preprocesamiento: Redimensionar y normalizar
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (80, 80))  # Dimensiones requeridas por el modelo
        image_normalized = np.array(image_resized, dtype=np.float32) / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)  # Expandir dimensiones para el modelo

        # Hacer la predicción
        predictions = model.predict(image_input)[0]  # Obtener la primera fila de resultados
        predicted_class_index = np.argmax(predictions)
        predicted_class = categories[predicted_class_index]

        # Mostrar las 3 clases más probables
        top_indices = np.argsort(predictions)[::-1][:3]
        print("\nTop 3 predicciones:")
        for i in top_indices:
            print(f"Clase: {categories[i]}, Confianza: {predictions[i]:.2f}")

        # Mostrar la imagen y la predicción principal
        plt.imshow(image)
        plt.title(f"Predicción: {predicted_class} (Confianza: {predictions[predicted_class_index]:.2f})")
        plt.axis('off')
        plt.show()

        print(f"Predicción final: {predicted_class}")
    except Exception as e:
        print(f"Error procesando la imagen: {e}")

# Ejemplo de uso
image_path = "/Users/alanmoreno/Desktop/TrabajosIA/Carros/pruebas/lambo12.jpg"
predict_car_image(image_path)
