from __future__ import annotations

import re

from tensorflow.python.keras.backend import constant
from tensorflow.python.types.core import Tensor
import tensorflow as tf

CLASS_PRIORITY: int = 900
CONJUNCTION_PRIORITY: int = 100
DEFAULT_PRIORITY: int = -1
DISEQUAL_PRIORITY: int = 200
DISJUNCTION_PRIORITY: int = 100
EQUIVALENCE_PRIORITY: int = 200
GREATER_EQUAL_PRIORITY: int = 200
GREATER_PRIORITY: int = 200
IMPLICATION_PRIORITY: int = 0
LESS_EQUAL_PRIORITY: int = 200
LESS_PRIORITY: int = 200
NEGATION_PRIORITY: int = 800
PAR_PRIORITY: int = 2000
VARIABLE_PRIORITY: int = 1000

DEFAULT_NAME: str = 'Abstract logic operator'
CLASS_X_NAME: str = 'Class X'
CLASS_Y_NAME: str = 'Class y'
CONJUNCTION_NAME: str = 'Conjunction operator'
DISEQUAL_NAME: str = 'Disequal operator'
DISJUNCTION_NAME: str = 'Disjunction operator'
EQUIVALENCE_NAME: str = 'Equivalence operator'
GREATER_EQUAL_NAME: str = 'Greater or equal operator'
GREATER_NAME: str = 'Greater operator'
IMPLICATION_NAME: str = 'Implication operator'
LESS_EQUAL_NAME: str = 'Less or equal operator'
LESS_NAME: str = 'Less operator'
LT_EQUIVALENCE_NAME: str = 'Logic tensor equivalence name'
NEGATION_NAME: str = 'Negation operator'
VARIABLE_NAME: str = 'Variable'

DEFAULT_REGEX: str = ''
CLASS_X_REGEX: str = 'X'
CLASS_Y_REGEX: str = '[a-z]+[0-9]*'
CONJUNCTION_REGEX: str = r'\^'
DISEQUAL_REGEX: str = '!='
DISJUNCTION_REGEX: str = r'\∨'  # this is a descending wedge not a v!
EQUIVALENCE_REGEX: str = '='
GREATER_EQUAL_REGEX: str = '>='
GREATER_REGEX: str = '>'
IMPLICATION_REGEX: str = '->'
LEFT_PAR_REGEX: str = r'\('
LESS_EQUAL_REGEX: str = '<='
LESS_REGEX: str = '<'
LT_EQUIVALENCE_REGEX: str = r'\|='
NEGATION_REGEX: str = r'\~'
RIGHT_PAR_REGEX: str = r'\)'
VARIABLE_REGEX: str = '[A-Z]([a-z]|[A-Z]|[0-9])+'


class LogicOperator:

    arity: int = 0
    priority: int = DEFAULT_PRIORITY

    def __init__(self, name: str = DEFAULT_NAME):
        self.name: str = name

    def accept(self):
        raise Exception('Try to call an abstract method')

    @staticmethod
    def parse(string: str):
        raise Exception('Try to call an abstract method')

    @staticmethod
    def _parse(regex: str, string: str) -> tuple[bool, str]:
        match = re.match(pattern=regex, string=string)
        return (True, string[match.span()[0]:match.span()[1]]) if match is not None and match.pos == 0 else (False, '')


class Op2(LogicOperator):

    arity = 2

    def __init__(self, l1: L, l2: L, name: str = DEFAULT_NAME):
        super().__init__(name)
        self.l1 = l1
        self.l2 = l2


class Op1(LogicOperator):

    arity = 1

    def __init__(self, name: str = DEFAULT_NAME):
        super().__init__(name)


class LeftPar(LogicOperator):

    priority: int = PAR_PRIORITY

    @staticmethod
    def parse(string: str):
        return LogicOperator._parse(LEFT_PAR_REGEX, string)


class RightPar(LogicOperator):

    priority: int = PAR_PRIORITY

    @staticmethod
    def parse(string: str):
        return LogicOperator._parse(RIGHT_PAR_REGEX, string)


class L(Op1):

    arity: int = 0
    priority: int = VARIABLE_PRIORITY

    def __init__(self, x: Tensor):
        """
        Logic variable, 0 is true and 1 is false.
        :param x: is a tensorflow tensor of one element (can be interpreted as a scalar)
        """

        super().__init__(VARIABLE_NAME)
        self.x = x

    def accept(self) -> L:
        return self

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(VARIABLE_REGEX, string)

    @staticmethod
    def reverse_relu(x: Tensor):
        return tf.maximum(x, L.true())

    @staticmethod
    def relu(x: Tensor):
        return tf.minimum(L.false(), x)

    @staticmethod
    def fringe(x: Tensor):
        return L.relu(L.reverse_relu(x))

    @staticmethod
    def true() -> Tensor:
        return constant(0)

    @staticmethod
    def false() -> Tensor:
        return constant(1)


class Equivalence(Op2):

    priority: int = EQUIVALENCE_PRIORITY

    def __init__(self, l1: L, l2: L):
        """
        Logic equivalence between two variables (x = y).
        """
        super().__init__(l1, l2, EQUIVALENCE_NAME)

    def accept(self) -> Tensor:
        return L.relu(tf.abs(self.l1.x - self.l2.x))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(EQUIVALENCE_REGEX, string)


class Implication(Op2):

    priority: int = IMPLICATION_PRIORITY

    def __init__(self, l1: L, l2: L):
        """
        Logic implication between two variable (x -> y)

        x | y | r
        0 | 0 | 0
        0 | 1 | 1
        1 | 0 | 0
        1 | 1 | 0
        """
        super().__init__(l1, l2, IMPLICATION_NAME)

    def accept(self) -> Tensor:
        return L.relu(self.l2.x - self.l1.x)

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(IMPLICATION_REGEX, string)


class Negation(Op1):

    priority: int = NEGATION_PRIORITY

    def __init__(self, l1: L):
        """
        Logic negation of a variable.
        """
        super().__init__(NEGATION_NAME)
        self.l1 = l1

    def accept(self) -> Tensor:
        return L.false() - L.fringe(self.l1.x)

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(NEGATION_REGEX, string)


class Conjunction(Op2):

    priority: int = CONJUNCTION_PRIORITY

    def __init__(self, l1: L, l2: L):
        """
        Logic conjunction between two variables (x ^ y).
        """
        super().__init__(l1, l2, CONJUNCTION_NAME)

    def accept(self) -> Tensor:
        return tf.maximum(self.l1.x, self.l2.x)

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(CONJUNCTION_REGEX, string)


class Disjunction(Op2):

    priority: int = DISJUNCTION_PRIORITY

    def __init__(self, l1: L, l2: L):
        """
        Logic disjunction between two variables (x ∨ y, this is a descending wedge not a v!).
        """
        super().__init__(l1, l2, DISJUNCTION_NAME)

    def accept(self) -> Tensor:
        return tf.minimum(self.l1.x, self.l2.x)

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(DISJUNCTION_REGEX, string)


class GreaterEqual(Op2):

    priority: int = GREATER_EQUAL_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, GREATER_EQUAL_NAME)

    def accept(self) -> Tensor:
        return L.relu(L.false() - L.relu(self.l1.x - self.l2.x))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(GREATER_EQUAL_REGEX, string)


class Greater(Op2):

    priority: int = GREATER_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, GREATER_NAME)

    def accept(self) -> Tensor:
        return tf.minimum(GreaterEqual(self.l1, self.l2).accept(), Disequal(self.l1, self.l2).accept())

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(GREATER_REGEX, string)


class Disequal(Op2):

    priority: int = DISEQUAL_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, DISEQUAL_NAME)

    def accept(self) -> Tensor:
        return L.false() - L.fringe(self.l1.x - self.l2.x)

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(DISEQUAL_REGEX, string)


class Less(Op2):

    priority: int = LESS_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, LESS_NAME)

    def accept(self) -> Tensor:
        return Conjunction(L(GreaterEqual(self.l1, self.l2).accept()), L(Disequal(self.l1, self.l2).accept())).accept()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(LESS_REGEX, string)


class LessEqual(Op2):

    priority: int = LESS_EQUAL_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, LESS_EQUAL_NAME)

    def accept(self) -> Tensor:
        return Negation(L(Greater(self.l1, self.l2).accept())).accept()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(LESS_EQUAL_REGEX, string)


class LT(L):

    arity: int = 0
    priority: int = CLASS_PRIORITY

    def __init__(self, x: Tensor):
        """
        :param x: is a 1D tensorflow tensor
        """
        super().__init__(x)

    def accept(self) -> LT:
        return self


class LTX(LT):

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(CLASS_X_REGEX, string)


class LTY(LT):

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(CLASS_Y_REGEX, string)


class LTEquivalence(Op2):

    priority: int = EQUIVALENCE_PRIORITY

    def __init__(self, l1: LT, l2: LT):
        """
        Element wise logic equivalence between the two tensors.
        """
        super().__init__(l1, l2, LT_EQUIVALENCE_NAME)

    def accept(self) -> Tensor:
        """
        :return: 'the most false value' (the maximum) among the partial results
        """
        xy = tf.stack([self.l1.x, self.l2.x], axis=1)
        element_wise_equivalence = tf.map_fn(lambda x: Equivalence(L(x[0]), L(x[1])).accept(), xy)
        return tf.reduce_max(element_wise_equivalence)

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(LT_EQUIVALENCE_REGEX, string)
