#!/usr/bin/python3

import keras
import _pickle as pickle
from keras import backend as K
from keras import optimizers
from keras import regularizers
from keras.callbacks import LearningRateScheduler
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


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    elif epoch > 100:
        lrate = 0.0003        
    return lrate


label_mode = 'fine'
dirname = 'cifar-100-python'
fpath = os.path.join(os.getcwd(), dirname, 'train')
x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')
fpath = os.path.join(os.getcwd(), dirname, 'test')
x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255.
x_test /= 255.

y_train = keras.utils.to_categorical(y_train, nb_classes)
y_test = keras.utils.to_categorical(y_test, nb_classes)

_, img_channels, img_rows, img_cols = x_train.shape

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

generator = ImageDataGenerator(rotation_range=90, width_shift_range=0.1,
                               height_shift_range=0.1, horizontal_flip=True)
generator.fit(x_train, seed=0)    

weight_decay = 1e-4
    
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=input_shape))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(nb_classes, activation='softmax'))

model.summary()

lrate = 0.01
decay_rate = lrate/nb_epoch
sgd = optimizers.SGD(lr=lrate, decay=decay_rate, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1,
          validation_data=(x_test, y_test), callbacks=[LearningRateScheduler(lr_schedule)])
score = model.evaluate(x_test, y_test, verbose=1)
print('\nTest result: %.3f loss: %.3f' % (score[1]*100,score[0]))
model.save(os.path.join(os.getcwd(), 'keras_cifar100_trained_model.h5'))
