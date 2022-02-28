import unittest
import numpy as np
from antlr4 import CommonTokenStream, InputStream
from tensorflow import constant
from tensorflow.python.ops.array_ops import gather, gather_nd
from tensorflow.python.ops.numpy_ops import argmax

from psyki.datalog import Fuzzifier
import tensorflow as tf
from resources.dist.resources.DatalogLexer import DatalogLexer
from resources.dist.resources.DatalogParser import DatalogParser
from test import POKER_FEATURE_MAPPING, POKER_CLASS_MAPPING
from test.resources import get_dataset, get_list_rules

true = tf.tile(tf.reshape(constant(0.), [1, 1]), [1, 1])
false = tf.tile(tf.reshape(constant(1.), [1, 1]), [1, 1])

fuzzifier = Fuzzifier(POKER_CLASS_MAPPING, POKER_FEATURE_MAPPING)
rules = get_list_rules('poker')
formulae = {rule: DatalogParser(CommonTokenStream(DatalogLexer(InputStream(rule)))) for rule in rules}
for _, rule in formulae.items():
    fuzzifier.visit(rule.formula())


class TestDatalog(unittest.TestCase):

    def test_nothing(self):
        hand1 = tf.constant([2, 6, 2, 1, 4, 13, 2, 4, 4, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 3, 10, 4, 7, 4, 9, 3, 8], dtype=tf.float32)
        output1 = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = fuzzifier.classes['nothing']

        # self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)
        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_two(self):
        hand1 = tf.constant([4, 9, 2, 2, 4, 2, 4, 6, 3, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 2, 2, 4, 7, 4, 10, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = fuzzifier.classes['two']

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_flush(self):
        hand1 = tf.constant([4, 4, 4, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        hand2 = tf.constant([4, 4, 1, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        function = fuzzifier.classes['flush']

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_full(self):
        hand1 = tf.constant([3, 2, 1, 2, 3, 11, 1, 11, 4, 11], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 4, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = fuzzifier.classes['full']

        # self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)
        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_four(self):
        hand1 = tf.constant([4, 9, 1, 9, 4, 7, 2, 9, 3, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 4, 5, 4, 7, 2, 9, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = fuzzifier.classes['four']

        # self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)
        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_three(self):
        hand1 = tf.constant([4, 9, 4, 2, 4, 7, 3, 9, 1, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 1, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        function = fuzzifier.classes['three']

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_pair(self):
        hand1 = tf.constant([4, 9, 4, 2, 4, 7, 4, 6, 2, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 2, 9], dtype=tf.float32)
        output1 = tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = fuzzifier.classes['pair']

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_straight(self):
        hand1 = tf.constant([1, 9, 4, 10, 2, 7, 4, 6, 3, 8], dtype=tf.float32)
        hand2 = tf.constant([1, 1, 4, 2, 2, 7, 4, 10, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        function = fuzzifier.classes['straight']

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

        # Straight is also 10, 11, 12, 13, 1!
        hand3 = tf.constant([1, 1, 4, 11, 2, 13, 4, 10, 3, 12], dtype=tf.float32)
        hand3 = tf.tile(tf.reshape(hand3, [1, 10]), [1, 1])
        output1 = tf.tile(tf.reshape(output1, [1, 10]), [1, 1])
        result = function(hand3, output1)
        tf.assert_equal(result, true)

    def test_straight_flush(self):
        hand1 = tf.constant([4, 9, 4, 10, 4, 7, 4, 6, 4, 8], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 3, 10, 4, 7, 4, 6, 3, 8], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        function = fuzzifier.classes['straight_flush']

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_royal_flush(self):
        hand1 = tf.constant([1, 10, 1, 11, 1, 13, 1, 12, 1, 1], dtype=tf.float32)
        hand2 = tf.constant([1, 9, 1, 11, 1, 13, 1, 10, 1, 12], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32)
        function = fuzzifier.classes['royal_flush']

        # self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)
        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def _test_reverse_implication_hand_output_combinations(self, function, hand1, hand2, output1, output2) -> None:
        result1, result2, result3, result4 = self._get_combination_values(function, hand1, hand2, output1, output2)
        tf.assert_equal(result1, true)
        tf.assert_equal(result2, true)
        tf.assert_equal(result3, false)
        tf.assert_equal(result4, true)

    def _test_double_implication_hand_output_combinations(self, function, hand1, hand2, output1, output2) -> None:
        result1, result2, result3, result4 = self._get_combination_values(function, hand1, hand2, output1, output2)
        tf.assert_equal(result1, true)
        tf.assert_equal(result2, false)
        tf.assert_equal(result3, false)
        tf.assert_equal(result4, true)

    @staticmethod
    def _get_combination_values(function, hand1, hand2, output1, output2):
        hand1 = tf.tile(tf.reshape(hand1, [1, 10]), [1, 1])
        hand2 = tf.tile(tf.reshape(hand2, [1, 10]), [1, 1])
        output1 = tf.tile(tf.reshape(output1, [1, 10]), [1, 1])
        output2 = tf.tile(tf.reshape(output2, [1, 10]), [1, 1])
        result1 = tf.reshape(function(hand1, output1), [1, 1])
        result2 = tf.reshape(function(hand2, output1), [1, 1])
        result3 = tf.reshape(function(hand1, output2), [1, 1])
        result4 = tf.reshape(function(hand2, output2), [1, 1])
        return result1, result2, result3, result4


class TestDatalogOnDataset(unittest.TestCase):

    def test_datalog(self):
        poker_training = get_dataset('poker-training')
        functions = [(name, fuzzifier.classes[name]) for name, _ in sorted(POKER_CLASS_MAPPING.items(), key=lambda i: i[1])]
        train_x = poker_training[:, :-1]
        train_y = poker_training[:, -1]
        train_y = np.eye(10)[train_y.astype(int)]
        x, y = tf.cast(train_x, dtype=tf.float32), tf.cast(train_y, dtype=tf.float32)
        result = tf.stack([tf.reshape(function[1](x, y), [x.shape[0], ]) for function in functions], axis=1)
        indices = tf.stack([range(0, len(poker_training)), argmax(train_y, axis=1)], axis=1)
        tf.assert_equal(gather_nd(result, indices), 0.)


if __name__ == '__main__':
    unittest.main()
