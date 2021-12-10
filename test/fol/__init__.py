import unittest
from psyki.fol import Parser
from psyki.fol.ast import AST
from psyki.logic import *
import tensorflow as tf


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
    rules = {
        'nothing': '',
        'pair': '',
        'twoPairs': '',
        'tris': '(R1 = R2 ^ R1 = R3) ∨ (R1 = R2 ^ R1 = R4) ∨ (R1 = R2 ^ R1 = R5) ∨ (R2 = R3 ^ R2 = R4) ∨ '
                '(R2 = R3 ^ R2 = R5) ∨ (R3 = R4 ^ R3 = R5) -> X |= tris',
        'straight': '',
        'flush': 'S1 = S2 ^ S1 = S3 ^ S1 = S4 -> X |= flush',
        'full': '',
        'poker': '(R1 = R2 ^ R1 = R3 ^ R1 = R4) ∨ (R1 = R2 ^ R1 = R3 ^ R1 = R5) ∨ '
                 '(R1 = R2 ^ R1 = R4 ^ R1 = R5) ∨ (R1 = R3 ^ R1 = R4 ^ R1 = R5) ∨ '
                 '(R2 = R3 ^ R2 = R4 ^ R2 = R5) -> X |= poker',
        'straightFlush': '',
        'royalFlush': ''
    }

    def test_flush(self):
        hand = tf.constant([4, 4, 4, 13, 4, 7, 4, 11, 4, 1], dtype=tf.float32)
        output = tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication])
        function = self._get_function(parser, 'flush')
        result = function(hand, output)
        self.assertEqual(result, L.true())

    def test_poker(self):
        hand = tf.constant([4, 9, 4, 9, 4, 7, 4, 9, 4, 9], dtype=tf.float32)
        output = tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32)
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, Disjunction, LeftPar, RightPar])
        function = self._get_function(parser, 'poker')
        result = function(hand, output)
        self.assertEqual(result, L.true())

    def test_tris(self):
        hand = tf.constant([4, 9, 4, 9, 4, 7, 4, 9, 4, 9], dtype=tf.float32)
        output = tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32)
        parser = Parser(
            [L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, Disjunction, LeftPar, RightPar])
        function = self._get_function(parser, 'tris')
        result = function(hand, output)
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
