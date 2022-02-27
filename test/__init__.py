import os
from os.path import isdir, dirname
from pathlib import Path
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from tensorflow.keras.layers import Dense
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import Tensor
from tensorflow.keras import Model
from test.experiments import statistics
from test.experiments import models
from test.resources import get_rules, get_dataset

POKER_FEATURE_MAPPING = {
        'S1': 0,
        'R1': 1,
        'S2': 2,
        'R2': 3,
        'S3': 4,
        'R3': 5,
        'S4': 6,
        'R4': 7,
        'S5': 8,
        'R5': 9
    }

POKER_CLASS_MAPPING = {
        'nothing': 0,
        'pair': 1,
        'two': 2,
        'three': 3,
        'straight': 4,
        'flush': 5,
        'full': 6,
        'four': 7,
        'straight_flush': 8,
        'royal_flush': 9
    }


def class_accuracy(_model, _x, _y) -> list:

    """
    :param _model: a neural network
    :param _x: features' data set
    :param _y: ground truth
    :return: a list of pairs representing accuracy and occurrence of a class for each class
    """
    predicted_y = np.argmax(_model.predict(_x), axis=1)
    match = np.equal(predicted_y, _y)
    accuracy = []
    for i in range(_model.output.shape[1]):
        accuracy.append([sum(match[_y == i]) / sum(_y == i), sum(_y == i)])
    return accuracy


def f1(_model, _x, _y_true, average: str = 'macro') -> float:
    predicted_y = np.argmax(_model.predict(_x), axis=1)
    return f1_score(_y_true, predicted_y, average=average)


def get_processed_dataset(name: str, validation: float = 1.0):
    poker_training = get_dataset(name + '-training')
    poker_testing = get_dataset(name + '-testing')
    if validation < 1:
        _, poker_testing = train_test_split(poker_testing, test_size=validation, random_state=123,
                                            stratify=poker_testing[:, -1])
    train_x = poker_training[:, :-1]
    train_y = poker_training[:, -1]
    test_x = poker_testing[:, :-1]
    test_y = poker_testing[:, -1]

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([train_y])
    encoder.fit_transform([test_y])

    return train_x, train_y, test_x, test_y


def get_mlp(input: Tensor, output: int, layers: int, neurons: int, activation_function, last_activation_function):
    """
    Generate a NN with the given parameters
    """
    x = Dense(neurons, activation=activation_function, name='L_1')(input)
    for i in range(2, layers):
        x = Dense(neurons, activation=activation_function, name='L_' + str(i))(x)
    return Dense(output, activation=last_activation_function, name='L_' + str(layers))(x)


class CustomCallback(Callback):

    def __init__(self, file):
        super().__init__()
        self.file = file
        self.best = 0.

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > self.best:
            self.best = logs.get('val_accuracy')
            model = Model(inputs=self.model.input, outputs=self.model.layers[-3].output)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.save(self.file)


def train_network(network, train_x, train_y, test_x, test_y, batch_size: int, epochs: int, file: str = None, knowledge=False):
    extension = '.h5'
    if file is None:
        file = 'experiment' + str(len([name for name in os.listdir(statistics.PATH)]) - 1)
    if not isdir(dirname(statistics.PATH / file)):
        Path(dirname(statistics.PATH / file)).mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(str(statistics.PATH / file) + '.csv', append=False, separator=';')
    if not isdir(dirname(models.PATH / file)):
        Path(dirname(models.PATH / file)).mkdir(parents=True, exist_ok=True)
    model_checkpoint = ModelCheckpoint(filepath=str(models.PATH / file) + extension, monitor='val_accuracy', mode='max', save_best_only=True)
    if knowledge:
        model_checkpoint = CustomCallback(str(models.PATH / file) + extension)
    network.fit(train_x, train_y, validation_data=(test_x, test_y), verbose=1, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger, model_checkpoint])
