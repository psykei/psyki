import unittest
from psyki.fol import Parser
from psyki.fol.ast import AST
from psyki.fol.operators import *
import tensorflow as tf
from test.resources import get_rules


class TestFol(unittest.TestCase):

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

    def test_nothing(self):
        hand1 = tf.constant([4, 5, 4, 10, 3, 7, 2, 6, 1, 8], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 3, 10, 4, 7, 4, 9, 3, 8], dtype=tf.float32)
        output1 = tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, LeftPar, RightPar, Exist,
                         Disjunction, Plus, Negation, Numeric, Product])
        function = self._get_function(parser, 'nothing')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_two_pairs(self):
        hand1 = tf.constant([4, 9, 2, 2, 4, 2, 4, 6, 3, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 2, 2, 4, 7, 4, 10, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Disjunction, Implication, LeftPar,
                         RightPar, Exist, Disequal])
        function = self._get_function(parser, 'twoPairs')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_flush(self):
        hand1 = tf.constant([4, 4, 4, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        hand2 = tf.constant([4, 4, 1, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication])
        function = self._get_function(parser, 'flush')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_full(self):
        hand1 = tf.constant([4, 9, 4, 2, 3, 9, 2, 9, 2, 2], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 4, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, Disjunction, LeftPar,
                         RightPar, Exist, Disequal])
        function = self._get_function(parser, 'full')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_poker(self):
        hand1 = tf.constant([4, 9, 1, 9, 4, 7, 2, 9, 3, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 4, 5, 4, 7, 2, 9, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser(
            [L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, Disjunction, LeftPar, RightPar])
        function = self._get_function(parser, 'poker')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_tris(self):
        hand1 = tf.constant([4, 9, 4, 2, 4, 7, 3, 9, 1, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 1, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        parser = Parser(
            [L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, Disjunction, LeftPar, RightPar, Exist])
        function = self._get_function(parser, 'tris')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_pair(self):
        hand1 = tf.constant([4, 9, 4, 2, 4, 7, 4, 6, 2, 9], dtype=tf.float32)
        hand2 = tf.constant([4, 1, 4, 2, 4, 7, 4, 10, 2, 9], dtype=tf.float32)
        output1 = tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, LeftPar, RightPar, Exist])
        function = self._get_function(parser, 'pair')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_straight(self):
        hand1 = tf.constant([1, 9, 4, 10, 2, 7, 4, 6, 3, 8], dtype=tf.float32)
        hand2 = tf.constant([1, 1, 4, 2, 2, 7, 4, 10, 3, 9], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, LeftPar, RightPar, Exist,
                         Disjunction, Plus, Negation, Numeric, Product])
        function = self._get_function(parser, 'straight')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

        # Straight is also 10, 11, 12, 13, 1!
        hand3 = tf.constant([1, 1, 4, 11, 2, 13, 4, 10, 3, 12], dtype=tf.float32)
        result = function(hand3, output1).get_value()
        self.assertEqual(result, L.true())

    def test_straight_flush(self):
        hand1 = tf.constant([4, 9, 4, 10, 4, 7, 4, 6, 4, 8], dtype=tf.float32)
        hand2 = tf.constant([4, 9, 3, 10, 4, 7, 4, 6, 3, 8], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, LeftPar, RightPar, Exist,
                         Disjunction, Plus, Negation, Numeric, Product])
        function = self._get_function(parser, 'straightFlush')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

    def test_royal_flush(self):
        hand1 = tf.constant([1, 1, 1, 11, 1, 13, 1, 10, 1, 12], dtype=tf.float32)
        hand2 = tf.constant([1, 9, 1, 11, 1, 13, 1, 10, 1, 12], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, LeftPar, RightPar, Exist,
                         Plus, Negation, Numeric])
        function = self._get_function(parser, 'royalFlush')

        self._test_hand_output_combinations(function, hand1, hand2, output1, output2)

    def _get_function(self, parser, label):
        label: str = self.rules[label]
        ops = parser.parse(label)
        ast = AST()
        for op in ops:
            ast.insert(op[0], op[1])
        return ast.root.call(self.input_mapping, self.output_mapping)

    def _test_hand_output_combinations(self, function, hand1, hand2, output1, output2) -> None:
        result1 = function(hand1, output1).get_value()
        result2 = function(hand2, output1).get_value()
        result3 = function(hand1, output2).get_value()
        result4 = function(hand2, output2).get_value()
        self.assertEqual(result1, L.true())
        self.assertEqual(result2, L.true())
        self.assertEqual(result3, L.false())
        self.assertEqual(result4, L.true())


if __name__ == '__main__':
    unittest.main()
