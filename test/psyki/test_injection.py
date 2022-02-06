import unittest
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Concatenate
from psyki import KnowledgeModule, Parser
from test import POKER_RULES, POKER_INPUT_MAPPING, get_dataset
import tensorflow as tf


class TestInjectionWithModules(unittest.TestCase):

    def test_modules_evaluation(self):
        net_input = Input((10,), name='Input')
        kms = KnowledgeModule.modules(POKER_RULES, Parser.extended_parser(), net_input, POKER_INPUT_MAPPING)
        network = Model(net_input, Concatenate(axis=1)(kms))

        poker_training = get_dataset('poker-training')
        train_x = poker_training[:, :-1]
        train_y = np.eye(10)[poker_training[:, -1].astype(int)]
        result = network.predict(train_x)
        result = np.apply_along_axis(lambda x: TestInjectionWithModules.zeros_except_max_value(x), axis=1, arr=result)
        tf.assert_equal(result, train_y)

    @staticmethod
    def zeros_except_max_value(row):
        max_index = len(row) - 1 - np.argmax(row[::-1])
        zeros = np.zeros(row.shape)
        zeros[max_index] = row[max_index]
        return zeros


if __name__ == '__main__':
    unittest.main()
