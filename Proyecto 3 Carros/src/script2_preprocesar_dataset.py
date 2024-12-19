import os
import cv2
import pickle
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

# Ruta donde se encuentra el conjunto de datos
data_dir = "dataset/"  # Cambia esto si tu carpeta de datos tiene otro nombre o ruta

# Lista de categorías (los nombres de las subcarpetas dentro de `data_dir`)
categories = ['bugatti_chiron', 'chevrolet_camaro', 'chevy_pop', 'lamborghini_aventador', 'mini_cooper']

# Función para modificar los canales RGB
def modify_rgb_channels(image, r_shift, g_shift, b_shift):
    """ Modificar los canales RGB de la imagen """
    b, g, r = cv2.split(image)  # Separar los canales de color
    r = cv2.add(r, r_shift)  # Aplicar desplazamiento al canal rojo
    g = cv2.add(g, g_shift)  # Aplicar desplazamiento al canal verde
    b = cv2.add(b, b_shift)  # Aplicar desplazamiento al canal azul
    return cv2.merge([b, g, r])  # Volver a combinar los canales

# Función para generar variantes con cambio en los canales RGB
def generate_rgb_variant(source_dir):
    img_names = os.listdir(source_dir)  # Procesar todas las imágenes
    r_shift, g_shift, b_shift = 10, 0, 0  # Cambiar solo el canal rojo

    for img_name in img_names:
        image_path = os.path.join(source_dir, img_name)
        if not os.path.isfile(image_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(image_path)
        modified_image = modify_rgb_channels(image, r_shift, g_shift, b_shift)
        augmented_image_name = f"{os.path.splitext(img_name)[0]}_rgb_variant.png"
        output_path = os.path.join(source_dir, augmented_image_name)

        if not os.path.exists(output_path):
            cv2.imwrite(output_path, modified_image)

# Función para aplicar opacidad a una imagen
def apply_opacity(image, alpha):
    overlay = np.zeros_like(image, dtype=np.uint8)  # Crear una capa negra
    return cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)

# Función para generar variantes con diferentes niveles de opacidad
def generate_opacity_variants(source_dir, alphas):
    img_names = os.listdir(source_dir)
    for img_name in img_names:
        image_path = os.path.join(source_dir, img_name)
        if not os.path.isfile(image_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        base_name, ext = os.path.splitext(img_name)
        for alpha in alphas:
            if alpha < 0.7:
                continue
            variant = apply_opacity(image, alpha)
            output_name = f"{base_name}_opacity{int(alpha * 100)}{ext}"
            output_path = os.path.join(source_dir, output_name)
            if not os.path.exists(output_path):
                cv2.imwrite(output_path, variant)

# Función para preprocesar una imagen
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (80, 80))  # Ajustar al tamaño correcto
    image = image / 255.0  # Normalizar los valores de los píxeles
    return image

# Función para generar más variantes para clases minoritarias
def augment_minority_classes(data_dir, categories, target_size):
    for category in categories:
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            continue

        # Número de imágenes actuales
        current_count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if current_count < target_size:
            additional_needed = target_size - current_count
            print(f"Generando {additional_needed} imágenes para la clase {category}")

            # Generar variantes
            generate_rgb_variant(path)
            generate_opacity_variants(path, [0.7, 0.8, 0.9, 1.0])

# Función para cargar y procesar los datos
def make_data():
    data = []

    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        if not os.path.exists(path):
            continue

        for img_name in os.listdir(path):
            image_path = os.path.join(path, img_name)
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image = preprocess_image(image_path)
            if image is not None:
                data.append([image, label])

    print(f"Total imágenes procesadas: {len(data)}")

    # Verificar balanceo de clases
    labels = [item[1] for item in data]
    class_counts = Counter(labels)
    print("Distribución de clases en el dataset:")
    for cls, count in class_counts.items():
        print(f"{categories[cls]}: {count}")

    # Guardar datos procesados
    with open('data/data.pickle', 'wb') as pik:
        pickle.dump(data, pik)

# Dividir los datos en entrenamiento y validación
def split_data():
    with open('data/data.pickle', 'rb') as pik:
        data = pickle.load(pik)

    X, y = zip(*data)
    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Entrenamiento: {len(X_train)}, Validación: {len(X_val)}")
    return X_train, X_val, y_train, y_val

# Ejecutar la función para generar el archivo
def main():
    target_size = max([len(os.listdir(os.path.join(data_dir, cat))) for cat in categories])
    augment_minority_classes(data_dir, categories, target_size)
    make_data()
    X_train, X_val, y_train, y_val = split_data()

if __name__ == "__main__":
    main()
