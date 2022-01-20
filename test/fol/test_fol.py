import unittest
import numpy as np
from psyki.fol.operators import *
import tensorflow as tf
from test import POKER_RULES
from test.resources import get_dataset


rule_index = {
        'nothing': 0,
        'pair': 1,
        'twoPairs': 2,
        'tris': 3,
        'straight': 4,
        'flush': 5,
        'full': 6,
        'poker': 7,
        'straightFlush': 8,
        'royalFlush': 9
    }
true = tf.tile(tf.reshape(L.true(), [1, 1]), [1, 1])
false = tf.tile(tf.reshape(L.false(), [1, 1]), [1, 1])


class TestFol(unittest.TestCase):

    def test_nothing(self):
        hand1 = tf.constant([2, 6, 2, 1, 4, 13, 2, 4, 4, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 3, 10, 4, 7, 4, 9, 3, 8], dtype=tf.float32)
        output1 = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['nothing']]

        # self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)
        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_two_pairs(self):
        hand1 = tf.constant([4, 9, 2, 2, 4, 2, 4, 6, 3, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 2, 2, 4, 7, 4, 10, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['twoPairs']]

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_flush(self):
        hand1 = tf.constant([4, 4, 4, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        hand2 = tf.constant([4, 4, 1, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['flush']]

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_full(self):
        hand1 = tf.constant([3, 2, 1, 2, 3, 11, 1, 11, 4, 11], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 4, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['full']]

        # self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)
        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)
    def test_poker(self):
        hand1 = tf.constant([4, 9, 1, 9, 4, 7, 2, 9, 3, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 4, 5, 4, 7, 2, 9, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['poker']]

        # self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)
        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_tris(self):
        hand1 = tf.constant([4, 9, 4, 2, 4, 7, 3, 9, 1, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 1, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['tris']]

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_pair(self):
        hand1 = tf.constant([4, 9, 4, 2, 4, 7, 4, 6, 2, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 2, 9], dtype=tf.float32)
        output1 = tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['pair']]

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_straight(self):
        hand1 = tf.constant([1, 9, 4, 10, 2, 7, 4, 6, 3, 8], dtype=tf.float32)
        hand2 = tf.constant([1, 1, 4, 2, 2, 7, 4, 10, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['straight']]

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

        # Straight is also 10, 11, 12, 13, 1!
        hand3 = tf.constant([1, 1, 4, 11, 2, 13, 4, 10, 3, 12], dtype=tf.float32)
        hand3 = tf.tile(tf.reshape(hand3, [1, 10]), [1, 1])
        output1 = tf.tile(tf.reshape(output1, [1, 10]), [1, 1])
        result = function(hand3, output1).get_value()
        tf.assert_equal(result, tf.tile(tf.reshape(L.true(), [1, 1]), [1, 1]))

    def test_straight_flush(self):
        hand1 = tf.constant([4, 9, 4, 10, 4, 7, 4, 6, 4, 8], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 3, 10, 4, 7, 4, 6, 3, 8], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['straightFlush']]

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_royal_flush(self):
        hand1 = tf.constant([1, 10, 1, 11, 1, 13, 1, 12, 1, 1], dtype=tf.float32)
        hand2 = tf.constant([1, 9, 1, 11, 1, 13, 1, 10, 1, 12], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32)
        function = POKER_RULES[rule_index['royalFlush']]

        # self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)
        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def _test_reverse_implication_hand_output_combinations(self, function, hand1, hand2, output1, output2) -> None:
        result1, result2, result3, result4 = self._get_combination_values(function, hand1, hand2, output1, output2)
        tf.assert_equal(result1, true)
        tf.assert_equal(result2, false)
        tf.assert_equal(result3, true)
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
        result1 = tf.reshape(function(hand1, output1).get_value(), [1, 1])
        result2 = tf.reshape(function(hand2, output1).get_value(), [1, 1])
        result3 = tf.reshape(function(hand1, output2).get_value(), [1, 1])
        result4 = tf.reshape(function(hand2, output2).get_value(), [1, 1])
        return result1, result2, result3, result4


class TestFolOnDataset(unittest.TestCase):

    def test_fol(self):
        poker_training = get_dataset('poker-training')
        functions = POKER_RULES
        train_x = poker_training[:, :-1]
        train_y = poker_training[:, -1]
        train_y = np.eye(10)[train_y.astype(int)]
        ten_zeros = tf.reshape(tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32), [1, 10])
        x, y = tf.cast(train_x, dtype=tf.float32), tf.cast(train_y, dtype=tf.float32)
        result = tf.stack([tf.reshape(function(x, y).get_value(), [x.shape[0], ]) for function in functions], axis=1)
        tf.assert_equal(result, tf.tile(ten_zeros, [train_x.shape[0], 1]))


if __name__ == '__main__':
    unittest.main()
