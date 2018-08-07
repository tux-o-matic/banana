__license__ = 'GPL-3.0'


import keras
import numpy as np
import os
import sys
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import Sequential
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


class Banana:

    def __init__(self):
        # Avoid warnings about CPU features
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    @staticmethod
    def get_model(img_channels, img_rows, img_cols, learning_rate, nb_classes, nb_epoch, wdecay=1e-4,
                  print_summary=False):
        """
        Return Keras Sequential CNN model.

        :param int img_channels: The number of channels in the image used for training, commonly 3
        :param int img_rows: The image width
        :param int img_cols: The image height
        :param float learning_rate: The learning rate
        :param int nb_classes: The number of classes to train the model for
        :param int nb_epoch: The number of epoch for the model training
        :param float wdecay: The weight decay for the regularizer
        :param bool print_summary: If model summary should be printed to stdout
        :return: Keras model
        :rtype: keras.models.Sequential
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(wdecay),
                         input_shape=(img_channels, img_rows, img_cols)))
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

        if print_summary:
            model.summary()

        decay_rate = learning_rate / nb_epoch
        sgd = optimizers.SGD(lr=learning_rate, decay=decay_rate, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        return model

    @staticmethod
    def train(dataset_test_dir, dataset_train_dir, nb_classes, steps_per_epoch, target_size, validation_steps,
              batch_size=32, learning_rate=0.01, log_dir=None, nb_epoch=100, trained_model='banana.h5', verbose=1):
        """
        Train Keras model with TensorFlow backend for image recognition.

        :param str dataset_test_dir: The directory containing the images to test against
        :param str dataset_train_dir: The directory containing the images for model training
        :param int nb_classes: The number of classes to train the model for
        :param int steps_per_epoch: The number of steps per epoch, ideally matches the number of training images
        :param tuple target_size: A tuple with the width and height of the pictures
        :param int validation_steps: The number of steps to run the validation, should match the number of test images
        :param int batch_size: The batch size (default: 32)
        :param float learning_rate: The starting learning rate, will vary as training progresses
        :param str log_dir: The directory to use for TensorBoard logging. If not specified, logging will be deactivate
        :param int nb_epoch: The number of epoch for the model training. Training will stop early if accuracy stagnates
        :param str trained_model: The full path of the trained Keras model to save
        :param bool verbose: If training progress should be printed to stdout
        """
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=90, width_shift_range=0.1,
                                           height_shift_range=0.1, horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(dataset_train_dir, target_size=target_size,
                                                            batch_size=batch_size, class_mode='binary')

        validation_generator = test_datagen.flow_from_directory(dataset_test_dir, target_size=target_size,
                                                                batch_size=batch_size, class_mode='binary')

        callbacks = [ReduceLROnPlateau(monitor='val_acc', factor=0.1, cooldown=0, patience=3, min_lr=1e-5,
                                       verbose=1),
                     EarlyStopping(monitor='val_acc', patience=10),
                     ModelCheckpoint(trained_model, monitor="val_acc", save_best_only=True, verbose=1)]
        if log_dir:
            callbacks.append(TensorBoard(log_dir=log_dir, histogram_freq=nb_epoch, batch_size=batch_size,
                                         write_grads=True))

        model = Banana.get_model(3, target_size[0], target_size[1], learning_rate, nb_classes, nb_epoch)

        model.fit_generator(train_generator, callbacks=callbacks, epochs=nb_epoch, steps_per_epoch=steps_per_epoch,
                            use_multiprocessing=True, verbose=verbose, workers=3, validation_data=validation_generator,
                            validation_steps=validation_steps)

        model.save(trained_model)