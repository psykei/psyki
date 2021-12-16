import unittest
from random import randint
import numpy as np
from psyki.fol import Parser
from psyki.fol.operators import *
import tensorflow as tf
from test.resources import get_rules, get_dataset, get_ordered_rules

input_mapping = {
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
output_mapping = {
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
rules = get_rules('poker')


class TestFol(unittest.TestCase):

    def test_nothing(self):
        hand1 = tf.constant([4, 5, 4, 10, 3, 7, 2, 6, 1, 8], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 3, 10, 4, 7, 4, 9, 3, 8], dtype=tf.float32)
        output1 = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, DoubleImplication, LeftPar, RightPar,
                         Exist, Disjunction, Plus, Negation, Numeric, Product, LessEqual])
        function = parser.get_function(rules['nothing'], input_mapping, output_mapping)

        self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_two_pairs(self):
        hand1 = tf.constant([4, 9, 2, 2, 4, 2, 4, 6, 3, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 2, 2, 4, 7, 4, 10, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Disjunction, ReverseImplication, LeftPar,
                         RightPar, Exist, Disequal])
        function = parser.get_function(rules['twoPairs'], input_mapping, output_mapping)

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_flush(self):
        hand1 = tf.constant([4, 4, 4, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        hand2 = tf.constant([4, 4, 1, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication])
        function = parser.get_function(rules['flush'], input_mapping, output_mapping)

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_full(self):
        hand1 = tf.constant([3, 2, 1, 2, 3, 11, 1, 11, 4, 11], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 4, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, DoubleImplication, Disjunction, LeftPar,
                         RightPar, Exist, Disequal])
        function = parser.get_function(rules['full'], input_mapping, output_mapping)

        self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_poker(self):
        hand1 = tf.constant([4, 9, 1, 9, 4, 7, 2, 9, 3, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 4, 5, 4, 7, 2, 9, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, DoubleImplication,
                         LeftPar, RightPar, Disjunction])
        function = parser.get_function(rules['poker'], input_mapping, output_mapping)

        self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_tris(self):
        hand1 = tf.constant([4, 9, 4, 2, 4, 7, 3, 9, 1, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 1, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, Disjunction, LeftPar,
                         RightPar, Exist])
        function = parser.get_function(rules['tris'], input_mapping, output_mapping)

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_pair(self):
        hand1 = tf.constant([4, 9, 4, 2, 4, 7, 4, 6, 2, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 2, 9], dtype=tf.float32)
        output1 = tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser(
            [L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar, Exist])
        function = parser.get_function(rules['pair'], input_mapping, output_mapping)

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_straight(self):
        hand1 = tf.constant([1, 9, 4, 10, 2, 7, 4, 6, 3, 8], dtype=tf.float32)
        hand2 = tf.constant([1, 1, 4, 2, 2, 7, 4, 10, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar, Exist,
                         Disjunction, Plus, Negation, Numeric, Product, LessEqual])
        function = parser.get_function(rules['straight'], input_mapping, output_mapping)

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

        # Straight is also 10, 11, 12, 13, 1!
        hand3 = tf.constant([1, 1, 4, 11, 2, 13, 4, 10, 3, 12], dtype=tf.float32)
        hand3 = tf.tile(tf.reshape(hand3, [1, 10]), [5, 1])
        output1 = tf.tile(tf.reshape(output1, [1, 10]), [5, 1])
        result = function(hand3, output1).get_value()
        tf.assert_equal(result, tf.tile(tf.reshape(L.true(), [1, 1]), [5, 1]))

    def test_straight_flush(self):
        hand1 = tf.constant([4, 9, 4, 10, 4, 7, 4, 6, 4, 8], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 3, 10, 4, 7, 4, 6, 3, 8], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar,
                         Exist, Disjunction, Plus, Negation, Numeric, Product, LessEqual])
        function = parser.get_function(rules['straightFlush'], input_mapping, output_mapping)

        self._test_reverse_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_royal_flush(self):
        hand1 = tf.constant([1, 10, 1, 11, 1, 13, 1, 12, 1, 1], dtype=tf.float32)
        hand2 = tf.constant([1, 9, 1, 11, 1, 13, 1, 10, 1, 12], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, DoubleImplication, LeftPar, RightPar,
                         Exist, Plus, Negation, Numeric])
        function = parser.get_function(rules['royalFlush'], input_mapping, output_mapping)

        self._test_double_implication_hand_output_combinations(function, hand1, hand2, output1, output2)

    def _test_reverse_implication_hand_output_combinations(self, function, hand1, hand2, output1, output2) -> None:
        result1, result2, result3, result4 = self._get_combination_values(function, hand1, hand2, output1, output2)
        true = tf.tile(tf.reshape(L.true(), [1, 1]), [5, 1])
        false = tf.tile(tf.reshape(L.false(), [1, 1]), [5, 1])
        tf.assert_equal(result1, true)
        tf.assert_equal(result2, false)
        tf.assert_equal(result3, true)
        tf.assert_equal(result4, true)

    def _test_double_implication_hand_output_combinations(self, function, hand1, hand2, output1, output2) -> None:
        result1, result2, result3, result4 = self._get_combination_values(function, hand1, hand2, output1, output2)
        true = tf.tile(tf.reshape(L.true(), [1, 1]), [5, 1])
        false = tf.tile(tf.reshape(L.false(), [1, 1]), [5, 1])
        tf.assert_equal(result1, true)
        tf.assert_equal(result2, false)
        tf.assert_equal(result3, false)
        tf.assert_equal(result4, true)

    @staticmethod
    def _get_combination_values(function, hand1, hand2, output1, output2):
        hand1 = tf.tile(tf.reshape(hand1, [1, 10]), [5, 1])
        hand2 = tf.tile(tf.reshape(hand2, [1, 10]), [5, 1])
        output1 = tf.tile(tf.reshape(output1, [1, 10]), [5, 1])
        output2 = tf.tile(tf.reshape(output2, [1, 10]), [5, 1])
        result1 = tf.reshape(function(hand1, output1).get_value(), [5, 1])
        result2 = tf.reshape(function(hand2, output1).get_value(), [5, 1])
        result3 = tf.reshape(function(hand1, output2).get_value(), [5, 1])
        result4 = tf.reshape(function(hand2, output2).get_value(), [5, 1])
        return result1, result2, result3, result4


class TestFolOnDataset(unittest.TestCase):

    def disabled_test_fol(self):
        ordered_rules = get_ordered_rules('poker')
        poker_training = get_dataset('poker-training')
        # random_indices = [randint(0, poker_training.shape[0] - 1) for _ in range(0, 1000)]
        random_indices = range(5, 17)
        poker_training = poker_training[random_indices, :]
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar,
                         Exist, Disjunction, Plus, Negation, Numeric, Product, Disequal, DoubleImplication, LessEqual])
        functions = [parser.get_function(rule, input_mapping, output_mapping) for rule in ordered_rules]
        train_x = poker_training[:, :-1]
        train_y = poker_training[:, -1]
        train_y = np.eye(10)[train_y.astype(int)]
        ten_zeros = tf.reshape(tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32), [1, 10])

        """for index, _ in enumerate(train_x):
            x, y = tf.constant(train_x[index, :], dtype=tf.float32), tf.constant(train_y[index, :], dtype=tf.float32)
            result = tf.concat([function(x, y).get_value() for function in functions], axis=0)
            if not all(tf.equal(result, ten_zeros)):
                print('Hand at index ' + str(index) + ' breaks the rules')
            self.assertTrue(all(tf.equal(result, ten_zeros)))"""

        x, y = tf.cast(train_x, dtype=tf.float32), tf.cast(train_y, dtype=tf.float32)
        result = tf.stack([tf.reshape(function(x, y).get_value(), [x.shape[0], ]) for function in functions], axis=1)
        tf.assert_equal(result, tf.tile(ten_zeros, [train_x.shape[0], 1]))


if __name__ == '__main__':
    unittest.main()
