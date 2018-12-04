'''
Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128 # The modele is trained by batch of 128 
num_classes = 10 # output classes : 10 numbers
epochs =  8

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
"""
    Nous allons ajouter au modele les couches de notre CNN
    Conv2D() : 1ere couche à convolution avec ReLu en fonction d'activation
    MaxPooling2D() : 1ere couche à convolution avec ReLu en fonction d'activation

"""
# 1st layer
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# 2eme layer
model.add(Conv2D(64, (3, 3), activation='relu'))

# Récupérer le max sur du 2x2: pool_height, pool_width: the size of the pooling window
model.add(MaxPooling2D(pool_size=(2, 2)))

"""
    Dropout consists in randomly setting a fraction rate of input units to 0 at each update 
    during training time, which helps prevent overfitting !
    The units that are kept are scaled by 1 / (1 - rate), so that their sum is unchanged at training time and inference time.
"""
model.add(Dropout(0.25))

model.add(Flatten()) # Flattens an input tensor while preserving the batch axis (axis 0).

"""
    This layer implements the operation: 
    outputs = activation(inputs * kernel + bias)  => CLASSICAL
    
    Where activation is the activation function passed as the activation argument (if not None), 
    kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only if use_bias is True by default).

    128 is 
"""

model.add(Dense(128, activation='relu'))


model.add(Dropout(0.5))

### Ajoute une sortie en softmax pour plusieurs probabilités
model.add(Dense(num_classes, activation='softmax'))

"""
    Hyperparameters with loss, optimizer
    # Compile the model withh loss function (crossentropy), optimizer (function of Stochastique Gradient with Adam Optimizer)
    metrics: juste accurency for precision on dataset 
"""

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])