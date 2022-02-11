import unittest
import numpy as np
from tensorflow.python.framework.random_seed import set_random_seed
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Concatenate
from psyki import StructuringInjector, ConstrainingInjector
from test import POKER_RULES, POKER_INPUT_MAPPING, get_dataset, get_mlp, POKER_OUTPUT_MAPPING
import tensorflow as tf


class TestInjectionWithModules(unittest.TestCase):
    poker_training = get_dataset('poker-training')
    train_x = poker_training[:, :-1]
    train_y = np.eye(10)[poker_training[:, -1].astype(int)]

    def test_modules_evaluation(self):
        net_input = Input((10,))
        injector = StructuringInjector(None)
        modules = injector.modules(POKER_RULES, net_input, POKER_INPUT_MAPPING)
        network = Model(net_input, Concatenate(axis=1)(modules))
        result = network.predict(self.train_x)
        result = np.apply_along_axis(lambda x: zeros_except_max_value(x), axis=1, arr=result)

        tf.assert_equal(result, self.train_y)

    def test_network_evaluation(self):
        set_random_seed(123)
        net_input = Input((10,))
        network = get_mlp(net_input, output=10, layers=3, neurons=128, hidden_activation='relu', last_activation='softmax')
        injector = StructuringInjector(network)
        injector.inject(POKER_RULES, 'softmax', POKER_INPUT_MAPPING)
        compile_fit(injector.predictor, self.train_x, self.train_y)
        accuracy = injector.predictor.evaluate(self.train_x, self.train_y)[1]

        self.assertTrue(accuracy > 0.99)


class TestInjectionWithConstraining(unittest.TestCase):
    poker_training = get_dataset('poker-training')
    train_x = poker_training[:, :-1]
    train_y = np.eye(10)[poker_training[:, -1].astype(int)]

    def test_network_evaluation(self):
        set_random_seed(123)
        net_input = Input((10,))
        network = get_mlp(net_input, output=10, layers=3, neurons=128, hidden_activation='relu', last_activation='softmax')
        injector = ConstrainingInjector(network)
        injector.inject(POKER_RULES, 'softmax', POKER_INPUT_MAPPING, POKER_OUTPUT_MAPPING)
        compile_fit(injector.predictor, self.train_x, self.train_y)
        injector.remove()
        compile(injector.predictor)
        accuracy = injector.predictor.evaluate(self.train_x, self.train_y)[1]

        self.assertTrue(accuracy > 0.60)


def zeros_except_max_value(row):
    max_index = len(row) - 1 - np.argmax(row[::-1])
    zeros = np.zeros(row.shape)
    zeros[max_index] = row[max_index]
    return zeros


def compile_fit(predictor, train_x, train_y) -> None:
    compile(predictor)
    predictor.fit(train_x, train_y, batch_size=32, epochs=10)


def compile(predictor) -> None:
    predictor.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])


if __name__ == '__main__':
    unittest.main()
