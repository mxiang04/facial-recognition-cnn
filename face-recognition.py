import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from matplotlib import image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from keras.utils import np_utils
import itertools

data = np.load('ORL_faces.npz')

x_train = data['trainX']

# preprocess the images with this thing / 255 for the pixels
x_train = np.array(x_train, dtype='float32') / 255

x_test = data['testX']
x_test = np.array(x_test, dtype='float32') / 255

y_train = data['trainY']
y_test = data['testY']

x_train, x_valid, y_train, y_valid = train_test_split(
    x_train, y_train, test_size=.05, random_state=1234,)

im_rows = 112
im_cols = 92
batch_size = 512
im_shape = (im_rows, im_cols, 1)

x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
x_valid = x_valid.reshape(x_valid.shape[0], *im_shape)

model = Sequential([
    Conv2D(filters=36, kernel_size=7, activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Conv2D(filters=54, kernel_size=5, activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(2024, activation='relu'),
    Dropout(0.5),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),

    # final layer
    Dense(20, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(lr=0.0001), metrics=['accuracy'])

history = model.fit(np.array(x_train), np.array(y_train), batch_size=512,
                    epochs=250, verbose=2, validation_data=(np.array(x_valid), np.array(y_valid)))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Facial recognition accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
