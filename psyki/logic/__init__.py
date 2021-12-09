from __future__ import annotations
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
VARIABLE_PRIORITY: int = 1000


class LogicOperation:

    def __init__(self):
        self.priority: int = DEFAULT_PRIORITY
        self.arity: int = 0

    def accept(self):
        pass


class Op2(LogicOperation):

    def __init__(self, l1: L, l2: L):
        super().__init__()
        self.arity: int = 2
        self.l1 = l1
        self.l2 = l2

    def accept(self):
        pass


class Op1(LogicOperation):

    def __init__(self):
        super().__init__()
        self.arity: int = 1

    def accept(self):
        pass


class L(Op1):

    def __init__(self, x: Tensor):
        """
        Logic variable, 0 is true and 1 is false.
        :param x: is a tensorflow tensor of one element (can be interpreted as a scalar)
        """
        super().__init__()
        self.x = x
        self.priority = VARIABLE_PRIORITY

    def accept(self) -> L:
        return self

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

    def __init__(self, l1: L, l2: L):
        """
        Logic equivalence between two variables (x = y).
        """
        super().__init__(l1, l2)
        self.priority = EQUIVALENCE_PRIORITY

    def accept(self) -> Tensor:
        return L.relu(self.l1.x - self.l2.x)


class Implication(Op2):

    def __init__(self, l1: L, l2: L):
        """
        Logic implication between two variable (x -> y)

        x | y | r
        0 | 0 | 0
        0 | 1 | 1
        1 | 0 | 0
        1 | 1 | 0
        """
        super().__init__(l1, l2)
        self.priority = IMPLICATION_PRIORITY

    def accept(self) -> Tensor:
        return L.relu(self.l1.x - self.l2.x)


class Negation(Op1):

    def __init__(self, l1: L):
        """
        Logic negation of a variable.
        """
        super().__init__()
        self.priority = NEGATION_PRIORITY
        self.l1 = l1

    def accept(self) -> Tensor:
        return L.false() - L.fringe(self.l1.x)


class Conjunction(Op2):

    def __init__(self, l1: L, l2: L):
        """
        Logic conjunction between two variables (x ^ y).
        """
        super().__init__(l1, l2)
        self.priority = CONJUNCTION_PRIORITY

    def accept(self) -> Tensor:
        return tf.maximum(self.l1.x, self.l2.x)


class Disjunction(Op2):

    def __init__(self, l1: L, l2: L):
        """
        Logic disjunction between two variables (x ∨ y, this is a descending wedge not a v!).
        """
        super().__init__(l1, l2)
        self.priority = DISJUNCTION_PRIORITY

    def accept(self) -> Tensor:
        return tf.minimum(self.l1.x, self.l2.x)


class GreaterEqual(Op2):

    def __init__(self, l1: L, l2: L):

        super().__init__(l1, l2)
        self.priority = GREATER_EQUAL_PRIORITY

    def accept(self) -> Tensor:
        return L.relu(L.false() - L.relu(self.l1.x - self.l2.x))


class Greater(Op2):

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2)
        self.priority = GREATER_PRIORITY

    def accept(self) -> Tensor:
        return tf.minimum(GreaterEqual(self.l1, self.l2).accept(), Disequal(self.l1, self.l2).accept())


class Disequal(Op2):

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2)
        self.priority = DISEQUAL_PRIORITY

    def accept(self) -> Tensor:
        return L.false() - L.fringe(self.l1.x - self.l2.x)


class Less(Op2):

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2)
        self.priority = LESS_PRIORITY

    def accept(self) -> Tensor:
        return Conjunction(L(GreaterEqual(self.l1, self.l2).accept()), L(Disequal(self.l1, self.l2).accept())).accept()


class LessEqual(Op2):

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2)
        self.priority = LESS_EQUAL_PRIORITY

    def accept(self) -> Tensor:
        return Negation(L(Greater(self.l1, self.l2).accept())).accept()


class LT(L):

    def __init__(self, x: Tensor):
        """
        :param x: is a 1D tensorflow tensor
        """
        super().__init__(x)

    def accept(self) -> LT:
        return self


class LTEquivalence(Op2):

    def __init__(self, l1: LT, l2: LT):
        """
        Element wise logic equivalence between the two tensors.

        :return: 'the most false value' (the maximum) among the partial results
        """
        super().__init__(l1, l2)
        self.priority = EQUIVALENCE_PRIORITY

    def accept(self) -> Tensor:
        xy = tf.stack([self.l1.x, self.l2.x], axis=1)
        element_wise_equivalence = tf.map_fn(lambda x: Equivalence(L(x[0]), L(x[1])).accept(), xy)
        return tf.reduce_max(element_wise_equivalence)
