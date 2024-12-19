import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, LeakyReLU, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Ruta donde se encuentra el conjunto de datos
data_dir = "dataset/"  # Cambia esto si tu carpeta de datos tiene otro nombre o ruta

# Lista de categorías (los nombres de las subcarpetas dentro de `data_dir`)
categories = ['bugatti_chiron', 'chevrolet_camaro', 'chevy_pop', 'lamborghini_aventador', 'mini_cooper']

# Función para preprocesar una imagen
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (80, 80))  # Ajustar al tamaño correcto
    image = image / 255.0  # Normalizar los valores de los píxeles
    return image

# Función para cargar los datos preprocesados
def load_data():
    with open('data/data.pickle', 'rb') as pik:
        data = pickle.load(pik)

    features, labels = zip(*data)
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return features, labels

# Dividir los datos en entrenamiento y validación
def split_data():
    features, labels = load_data()
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    print(f"Entrenamiento: {len(X_train)}, Validación: {len(X_val)}")
    return X_train, X_val, y_train, y_val

# Crear y compilar el modelo
def create_model(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    conv1 = Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same')(input_layer)
    act1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D((2, 2), padding='same')(act1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same')(drop1)
    act2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D((2, 2), padding='same')(act2)
    drop2 = Dropout(0.5)(pool2)

    flt = Flatten()(drop2)
    dense1 = Dense(128, activation='linear')(flt)
    act3 = LeakyReLU(alpha=0.1)(dense1)
    drop3 = Dropout(0.5)(act3)

    out = Dense(num_classes, activation='softmax')(drop3)

    model = Model(inputs=input_layer, outputs=out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Entrenar el modelo
def train_model(model, train_dataset, val_dataset, epochs=10):
    from tensorflow.keras.callbacks import EarlyStopping

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    model.save('./modelo/mymodel.keras', save_format='keras')
    with open('history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    print("Modelo y historial guardados con éxito.")
    return history

# Visualizar resultados de entrenamiento
def visualize_training(history):
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(accuracy))

    # Graficar precisión
    plt.figure()
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Graficar pérdida
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# Evaluar el modelo
def evaluate_model():
    model = tf.keras.models.load_model('./modelo/mymodel.keras')
    X_train, X_val, y_train, y_val = split_data()
    predictions = model.predict(X_val)
    predicted_classes = np.argmax(predictions, axis=1)

    target_names = categories
    print(classification_report(y_val, predicted_classes, target_names=target_names))

    # Matriz de confusión
    cm = confusion_matrix(y_val, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Visualizar predicciones
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_val[i])
        plt.xlabel(f'Real: {categories[y_val[i]]}\nPred: {categories[predicted_classes[i]]}')
        plt.xticks([])
        plt.yticks([])
    plt.show()

# Pipeline principal
def main():
    X_train, X_val, y_train, y_val = split_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32).prefetch(tf.data.AUTOTUNE)

    model = create_model((80, 80, 3), len(categories))
    model.summary()

    history = train_model(model, train_dataset, val_dataset, epochs=10)

    with open('history.pkl', 'rb') as f:
        history_data = pickle.load(f)
    visualize_training(history_data)

    evaluate_model()

if __name__ == "__main__":
    main()
