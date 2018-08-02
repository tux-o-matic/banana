#!/usr/bin/python3

import keras
import _pickle as pickle
import numpy as np
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
import os
import sys

# Avoid warnings about CPU features
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 128
nb_classes = 100
nb_epoch = 100
saved_model = os.path.join(os.getcwd(), 'keras_cifar100_trained_model.h5')


def load_batch(fpath, label_key='labels'):
    with open(fpath, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded           
    data = d['data']
    labels = d[label_key]
    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


label_mode = 'fine'
dirname = 'cifar-100-python'
fpath = os.path.join(os.getcwd(), dirname, 'train')
x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')
fpath = os.path.join(os.getcwd(), dirname, 'test')
x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

x_train = x_train.astype('float32') / 255.0 - 0.5
x_test = x_test.astype('float32') / 255.0 - 0.5

y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)

_, img_channels, img_rows, img_cols = x_train.shape

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)

generator = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
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

board = TensorBoard(log_dir='./logs', histogram_freq=x_train.shape[0], batch_size=batch_size, write_grads=True)
lr_reducer = ReduceLROnPlateau(monitor='val_acc', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=1e-5, verbose=1)
early_stopping_callback = EarlyStopping(monitor='val_acc', patience=10)
model_checkpoint = ModelCheckpoint(saved_model, monitor="val_acc", save_best_only=True, verbose=1)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit_generator(generator.flow(x_train, y_train, batch_size=batch_size), epochs=nb_epoch, verbose=1, workers=3,
                    use_multiprocessing=True, validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0]/nb_classes,
                    callbacks=[board, lr_reducer, early_stopping_callback, model_checkpoint])
score = model.evaluate(x_test, y_test, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (score[1]*100, score[0]))
model.save(saved_model)
