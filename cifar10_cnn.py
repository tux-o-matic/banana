#!/usr/bin/python3

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import keras
import numpy as np
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import cifar10
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import sys

# Avoid warnings about CPU features
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 129
nb_classes = 10
nb_epoch = 100
saved_model = os.path.join(os.getcwd(), 'keras_cifar10_trained_model.h5')


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0 - 0.5
x_test = x_test.astype('float32') / 255.0 - 0.5

y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)

_, img_channels, img_rows, img_cols = x_train.shape

generator = ImageDataGenerator(rotation_range=90, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
generator.fit(x_train, seed=0)    

wdecay = 1e-4
    
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wdecay), input_shape=x_train.shape[1:]))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wdecay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wdecay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wdecay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wdecay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wdecay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(nb_classes, activation='softmax'))

model.summary()

lrate = 0.001
decay_rate = lrate/nb_epoch
sgd = optimizers.SGD(lr=lrate, decay=decay_rate, momentum=0.9, nesterov=True)

lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-5, verbose=1)
early_stopping_callback = EarlyStopping(monitor='val_acc', patience=10)
model_checkpoint = ModelCheckpoint(saved_model, monitor="val_acc", save_best_only=True, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size), epochs=nb_epoch, verbose=1, workers=3,
                    use_multiprocessing=True, validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0]/nb_classes,
                    callbacks=[lr_reducer, early_stopping_callback, model_checkpoint])
score = model.evaluate(x_test, y_test, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (score[1]*100, score[0]))
model.save(saved_model)
