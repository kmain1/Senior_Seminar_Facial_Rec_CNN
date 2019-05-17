
# this is a large adaptation of kaggle user sivarajh's olivetti project
# https://www.kaggle.com/imrandude/olivetti/downloads/olivetti_faces_target.npy/comments
# as well as references from youtube user sendex
# https://www.youtube.com/watch?v=WvoLTXIjBYU&t=957s

import keras
import pickle
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

batch_size = 10
num_classes = 40
epochs = 100
IMG_SIZE = 64

data_imgs = np.load("C:\\Users\\Kyle\\Documents\\lfw-deepfunneled\\input\\olivetti_faces.npy")
labels = np.load("C:\\Users\\Kyle\\Documents\\lfw-deepfunneled\\input\\olivetti_faces_target.npy")
data = data_imgs.reshape(data_imgs.shape[0], data_imgs.shape[1] * data_imgs.shape[2])

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(0, 400):
    if(i % 10 == 8 or i % 10 == 9):
        x_test.append(data_imgs[i])
        y_test.append(labels[i])
    else:
        x_train.append(data_imgs[i])
        y_train.append(labels[i])

print(f"Length of x_test: {len(x_test)}")
print(f"Length of x_train: {len(x_train)}")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array(y_train)
y_test = np.array(y_test)


# to use no max pooling layers uncommment the strides in the current Conv Layers
#   Comment out model.add(MaxPooling2D)
#   Uncomment the replacement Conv Layers

model = Sequential()
model.add(Conv2D(64, 
                kernel_size=(3, 3), 
                #strides=(2,2), # have to increment preceding conv layer stride to remove pooling
                activation='relu',
                input_shape=(IMG_SIZE,IMG_SIZE,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, 
#                 kernel_size=(2,2),  #replacing the 2 x 2 pooling with a conv layer
#                 activation='relu',      #default pooling stride are the size of the filter (2 x 2)
#                 strides=(2,2)))
model.add(Conv2D(64, 
                kernel_size=(3, 3), 
                #strides=(2,2), # have to increment preceding conv layer stride to remove pooling
                activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, 
#                 kernel_size=(2,2),  #replacing the 2 x 2 pooling with a conv layer
#                 activation='relu',      #default pooling stride are the size of the filter (2 x 2)
#                 strides=(2,2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, 
                # uncomment which ever optimizer you want to use
                #optimizer=keras.optimizers.SGD(lr=0.001, momentum=1, decay=0.05,nesterov=True),
                optimizer=keras.optimizers.Adam(lr=0.001),
                metrics=['accuracy'])
history = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1, 
        validation_data=(x_test, y_test)
        )

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Adam model with pooling: accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Adam model with pooling: loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


