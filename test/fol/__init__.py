import unittest
from psyki.fol import Parser
from psyki.fol.ast import AST
from psyki.logic import Equivalence, Conjunction, Implication, L, LT, LTEquivalence, LTX, LTY
import tensorflow as tf


class TestFol(unittest.TestCase):

    def test_parse(self):
        string: str = 'S1 = S2 ^ S1 = S3 ^ S1 = S4 -> X |= suit.'
        input_mapping = {
            'S1': 0,
            'S2': 2,
            'S3': 4,
            'S4': 6,
            'S5': 8,
        }
        output_mapping = {
            'suit.': tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        }
        hand = tf.constant([4, 4, 4, 13, 4, 7, 4, 11, 4, 1])
        output = tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication])
        ops = parser.parse(string)
        ast = AST()
        for op in ops:
            ast.insert(op[0], op[1])
        function = ast.root.call(input_mapping, output_mapping)
        result = function(hand, output)
        print(result)
