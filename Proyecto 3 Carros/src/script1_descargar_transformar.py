import os
import cv2
import numpy as np
from bing_image_downloader import downloader

# === GENERADOR DE IMÁGENES ===
def make_background_transparent(image, threshold=0):
    tmp = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    black_pixels = np.all(tmp[:, :, :3] <= threshold, axis=2)
    tmp[black_pixels, 3] = 0
    return tmp

def descargar_y_transformar_imagenes(categorias, total_images, base_output_dir, image_size=(80, 80), num_variants=10):
    for query in categorias:
        print(f"Descargando imágenes para: {query}")
        output_dir = os.path.join(base_output_dir, query)
        os.makedirs(output_dir, exist_ok=True)

        try:
            downloader.download(query, limit=total_images, output_dir="temp", adult_filter_off=True)
        except Exception as e:
            print(f"Error al descargar imágenes para {query}: {e}")
            continue

        folder_path = os.path.join("temp", query)
        if not os.path.exists(folder_path):
            print(f"No se encontró la carpeta de descarga para {query}.")
            continue

        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"No se pudo cargar la imagen: {filename}")
                        continue

                    # Redimensionar imagen original
                    resized_original = cv2.resize(img, image_size)
                    original_variant_filename = f"{os.path.splitext(filename)[0]}_original.png"
                    resized_original_path = os.path.join(output_dir, original_variant_filename)
                    cv2.imwrite(resized_original_path, resized_original)

                    # Crear variantes rotadas
                    for i in range(num_variants):
                        angle = np.random.randint(0, 360)
                        M = cv2.getRotationMatrix2D((image_size[0] // 2, image_size[1] // 2), angle, 1)
                        rotated_variant = cv2.warpAffine(resized_original, M, image_size, borderMode=cv2.BORDER_CONSTANT)
                        rotated_variant = make_background_transparent(rotated_variant)

                        variant_filename = f"{os.path.splitext(filename)[0]}_variant{i}.png"
                        variant_path = os.path.join(output_dir, variant_filename)
                        cv2.imwrite(variant_path, rotated_variant)
                except Exception as e:
                    print(f"Error al procesar la imagen {filename}: {e}")

        # Eliminar carpeta temporal
        for file_name in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, file_name))
        os.rmdir(folder_path)
        print(f"Descarga y transformación completadas para {query}.")

# Lista de categorías
categories = ['porsche 911', 'fiat 500', 'hummer H1', 'volkswagen kombi', 'ford mustang']

# Descargar imágenes para cada categoría
descargar_y_transformar_imagenes(
    categorias=categories,
    total_images=100,  # Número de imágenes a descargar por categoría
    base_output_dir='dataset',
    image_size=(80, 80),
    num_variants=10
)
