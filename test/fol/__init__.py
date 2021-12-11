import unittest
from psyki.fol import Parser
from psyki.fol.ast import AST
from psyki.logic import *
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

    def test_flush(self):
        hand1 = tf.constant([4, 4, 4, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        output1 = tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication])
        function = self._get_function(parser, 'flush')
        result1 = function(hand1, output1).get_value()
        self.assertEqual(result1, L.true())

        hand2 = tf.constant([4, 4, 1, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        result2 = function(hand2, output1).get_value()
        self.assertEqual(result2, L.true())

        output2 = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        result3 = function(hand1, output2).get_value()
        self.assertEqual(result3, L.false())

    def test_poker(self):
        hand = tf.constant([4, 9, 4, 9, 4, 7, 4, 9, 4, 9], dtype=tf.float32)
        output = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, Disjunction, LeftPar, RightPar])
        function = self._get_function(parser, 'poker')
        result = function(hand, output).get_value()
        self.assertEqual(result, L.true())

    def test_tris(self):
        hand = tf.constant([4, 9, 4, 2, 4, 7, 4, 9, 4, 9], dtype=tf.float32)
        output = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser(
            [L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, Disjunction, LeftPar, RightPar, Exist])
        function = self._get_function(parser, 'tris')
        result = function(hand, output).get_value()
        self.assertEqual(result, L.true())

    def _get_function(self, parser, label):
        label: str = self.rules[label]
        ops = parser.parse(label)
        ast = AST()
        for op in ops:
            ast.insert(op[0], op[1])
        return ast.root.call(self.input_mapping, self.output_mapping)


if __name__ == '__main__':
    unittest.main()
