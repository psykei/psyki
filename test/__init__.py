import os
import numpy as np
from keras.callbacks import CSVLogger
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder

from psyki.fol import Parser
from psyki.fol.operators import *
from test.experiments import statistics
from test.experiments import models
from test.resources import get_rules, get_dataset

POKER_INPUT_MAPPING = {
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
POKER_OUTPUT_MAPPING = {
        'nothing':          tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32),
        'pair':             tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32),
        'twoPairs':         tf.constant([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32),
        'tris':             tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32),
        'straight':         tf.constant([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=tf.float32),
        'flush':            tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=tf.float32),
        'full':             tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32),
        'poker':            tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32),
        'straightFlush':    tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32),
        'royalFlush':       tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
    }
_parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar,
                 Exist, Disjunction, Plus, Negation, Numeric, Product, Disequal, DoubleImplication, LessEqual])
POKER_RULES = [_parser.get_function(rule, POKER_INPUT_MAPPING, POKER_OUTPUT_MAPPING)
               for _, rule in get_rules('poker').items()]


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


def get_processed_dataset(name: str):
    poker_training = get_dataset(name + '-training')
    poker_testing = get_dataset(name + '-testing')

    train_x = poker_training[:, :-1]
    train_y = poker_training[:, -1]
    test_x = poker_training[:, :-1]
    test_y = poker_training[:, -1]

    # One Hot encode the class labels
    encoder = OneHotEncoder(sparse=False)
    encoder.fit_transform([train_y])
    encoder.fit_transform([test_y])

    return train_x, train_y, test_x, test_y


def get_mlp(input: Tensor, output: int, layers: int, neurons: int, activation_function, last_activation_function):
    x = Dense(neurons, activation=activation_function, name='Layer_1')(input)
    for i in range(2, layers):
        x = Dense(neurons, activation=activation_function, name='Layer_' + str(i))(x)
    return Dense(output, activation=last_activation_function, name='Layer_' + str(layers))(x)


def train_network(network, train_x, train_y, test_x, test_y, batch_size: int, epochs: int, file: str = None):
    if file is None:
        file = 'experiment' + str(len([name for name in os.listdir(statistics.PATH)]) - 1)
    csv_logger = CSVLogger(str(statistics.PATH / file) + '.csv', append=False, separator=';')
    network.fit(train_x, train_y, validation_data=(test_x, test_y), verbose=1, batch_size=batch_size, epochs=epochs, callbacks=[csv_logger])


def save_network(network, file):
    if file is None:
        file = 'experiment' + str(len([name for name in os.listdir(statistics.PATH)]) - 1)
    network.save(str(models.PATH / file) + '.h5', save_format='h5')


def save_network_from_injector(injector, file):
    if file is None:
        file = 'experiment' + str(len([name for name in os.listdir(statistics.PATH)]) - 1)
    file = str(models.PATH / file) + '.h5'
    injector.save(file)
