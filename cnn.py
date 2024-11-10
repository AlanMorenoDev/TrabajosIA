import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU

dirname = os.path.join(os.getcwd(), 'sportimages')
imgpath = dirname + os.sep

images = []
directories = []
dircount = []
prevRoot = ''
cant = 0

print("Leyendo imágenes de ", imgpath)

for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search(r"\.(jpg|jpeg|png|bmp|tiff)$", filename):
            cant += 1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            if len(image.shape) == 3:
                images.append(image)
            print("Leyendo... ", cant, end="\r")
            if prevRoot != root:
                print(root, cant)
                prevRoot = root
                directories.append(root)
                dircount.append(cant)
                cant = 0
dircount.append(cant)
dircount = dircount[1:]
dircount[0] += 1

print('Directorios leídos:', len(directories))
print("Imágenes en cada directorio", dircount)
print('Suma total de imágenes en subdirs:', sum(dircount))

labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice += 1

print("Cantidad de etiquetas creadas: ", len(labels))

deportes = []
indice = 0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice, name[len(name) - 1])
    deportes.append(name[len(name) - 1])
    indice += 1


y = np.array(labels)
X = np.array(images, dtype=np.uint8)  


classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs:', nClasses)
print('Output classes:', classes)


train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2)
train_X = train_X.astype('float32') / 255.0
test_X = test_X.astype('float32') / 255.0


train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)


train_X, valid_X, train_label, valid_label = train_test_split(
    train_X, train_Y_one_hot, test_size=0.2, random_state=13
)


INIT_LR = 1e-3  
epochs = 20  
batch_size = 64  

sport_model = Sequential()
sport_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(train_X.shape[1], train_X.shape[2], 3)))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(MaxPooling2D((2, 2), padding='same'))
sport_model.add(Dropout(0.5))

sport_model.add(Flatten())
sport_model.add(Dense(32, activation='linear'))
sport_model.add(LeakyReLU(alpha=0.1))
sport_model.add(Dropout(0.5))
sport_model.add(Dense(nClasses, activation='softmax'))

sport_model.summary()

sport_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=INIT_LR, decay=INIT_LR / 100),
    metrics=['accuracy']
)

sport_train = sport_model.fit(
    train_X, train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label)
)

sport_model.save("/home/likcos/setDocto/sport.h5")

test_loss, test_accuracy = sport_model.evaluate(test_X, test_Y_one_hot)
print('Test accuracy:', test_accuracy)
