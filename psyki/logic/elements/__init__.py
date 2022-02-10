from __future__ import annotations
import re
from tensorflow.keras.backend import constant
from tensorflow.python.types.core import Tensor
import tensorflow as tf
from psyki.utils import abstract_method_exception, eta

CLASS_PRIORITY: int = 1000
CONJUNCTION_PRIORITY: int = 100
DEFAULT_PRIORITY: int = -1
NOT_EQUAL_PRIORITY: int = 200
DISJUNCTION_PRIORITY: int = 100
DOUBLE_IMPLICATION_PRIORITY: int = 0
EQUIVALENCE_PRIORITY: int = 200
EXIST_PRIORITY: int = 2000
GREATER_EQUAL_PRIORITY: int = 200
GREATER_PRIORITY: int = 200
IMPLICATION_PRIORITY: int = 0
LESS_EQUAL_PRIORITY: int = 200
LESS_PRIORITY: int = 200
NEGATION_PRIORITY: int = 300
NUMERIC_PRIORITY: int = 1000
PAR_PRIORITY: int = 5000
PLUS_PRIORITY: int = 400
PRODUCT_PRIORITY: int = 500
REVERSE_IMPLICATION_PRIORITY: int = 0
VARIABLE_PRIORITY: int = 1000

DEFAULT_REGEX: str = ''
CLASS_X_REGEX: str = 'X'
CLASS_Y_REGEX: str = '[a-z]+([A-Z]|[a-z]|[0-9])*'
CONJUNCTION_REGEX: str = r'\^'
NOT_EQUAL_REGEX: str = '!='
DISJUNCTION_REGEX: str = r'\∨'  # this is \vee not a v!
DOUBLE_IMPLICATION_REGEX: str = '<->'
EQUIVALENCE_REGEX: str = '='
GREATER_EQUAL_REGEX: str = '>='
GREATER_REGEX: str = '>(?!=)'
IMPLICATION_REGEX: str = '->'
LEFT_PAR_REGEX: str = r'\('
LESS_EQUAL_REGEX: str = '<='
LT_EQUIVALENCE_REGEX: str = r'\|='
NEGATION_REGEX: str = r'\~'
NUMERIC_REGEX: str = '[+-]?([0-9]*[.])?[0-9]+'
PLUS_REGEX: str = '[+]'
PRODUCT_REGEX: str = r'\*'
REVERSE_IMPLICATION_REGEX: str = r'<-(?!>)'
LESS_REGEX: str = r'<(?!-)(?!=)'
RIGHT_PAR_REGEX: str = r'\)'
SKIP_REGEX: str = r'/'
VARIABLE_REGEX: str = r'^(?!' + CLASS_X_REGEX + ')[A-Z]([a-z]|[A-Z])*[0-9]*'


class LogicElement:
    arity: int = 0
    priority: int = DEFAULT_PRIORITY

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def __eq__(self, other):
        return abstract_method_exception()

    def __hash__(self):
        return abstract_method_exception()

    def copy(self) -> LogicElement:
        return abstract_method_exception()

    def compute(self) -> LogicElement:
        return abstract_method_exception()

    @property
    def value(self) -> Tensor:
        return abstract_method_exception()

    @staticmethod
    def parse(string: str):
        return abstract_method_exception()

    @staticmethod
    def _parse(regex: str, string: str) -> tuple[bool, str]:
        match = re.match(pattern=regex, string=string)
        return (True, string[match.span()[0]:match.span()[1]]) if match is not None and match.pos == 0 else (False, '')


class LeftPar(LogicElement):
    priority: int = PAR_PRIORITY

    @staticmethod
    def parse(string: str):
        return LogicElement._parse(LEFT_PAR_REGEX, string)


class RightPar(LogicElement):
    priority: int = PAR_PRIORITY

    @staticmethod
    def parse(string: str):
        return LogicElement._parse(RIGHT_PAR_REGEX, string)


class Function2(LogicElement):

    arity: int = 2

    def __init__(self, l1: Variable, l2: Variable):
        super().__init__()
        self.l1 = l1
        self.l2 = l2

    def __eq__(self, other: Function2):
        return str(self) == str(other) and self.l1 == self.l1 and self.l2 == self.l2

    def __hash__(self):
        return hash(str(self)) + hash(self.l1) + hash(self.l2)

    def copy(self) -> LogicElement:
        return Function2(self.l1, self.l2)


class Function1(LogicElement):

    arity: int = 1

    def __init__(self, v: Variable):
        super().__init__()
        self.v = v

    def __eq__(self, other: Function1):
        return str(self) == str(other) and self.v == other.v

    def __hash__(self):
        return hash(str(self)) + hash(self.v)


class Skip(Function2):
    priority: int = IMPLICATION_PRIORITY

    def compute(self) -> Variable:
        return Variable(tf.zeros(tf.shape(self.l2.value)[0]))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(SKIP_REGEX, string)


class Variable(LogicElement):

    priority: int = VARIABLE_PRIORITY

    def __init__(self, x: Tensor):
        """
        Logic variable, 0 is true and 1 is false.
        :param x: is a tensorflow tensor of one element (can be interpreted as a scalar)
        """
        super().__init__()
        self.x = x

    def __hash__(self):
        return hash(str(self)) + self.x.__hash__()

    def copy(self) -> LogicElement:
        return Variable(self.x)

    def compute(self) -> Variable:
        return self

    @property
    def value(self) -> Tensor:
        return self.x

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(VARIABLE_REGEX, string)

    @staticmethod
    def true() -> Tensor:
        return constant(0)

    @staticmethod
    def false() -> Tensor:
        return constant(1)


class Constant(Variable):

    priority: int = NUMERIC_PRIORITY

    def __init__(self, x: str):
        """
        Logic variable, 0 is true and 1 is false.
        :param x: is a tensorflow tensor of one element (can be interpreted as a scalar)
        """

        super().__init__(tf.constant(float(x)))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(NUMERIC_REGEX, string)


class Equivalence(Function2):

    priority: int = EQUIVALENCE_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        """
        Logic equivalence between two variables (x = y).
        """
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Variable(eta(tf.abs(self.l1.x - self.l2.x)))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(EQUIVALENCE_REGEX, string)


class Implication(Function2):

    priority: int = IMPLICATION_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        """
        Logic implication between two variable (x -> y)

        x | y | r
        0 | 0 | 0
        0 | 1 | 1
        1 | 0 | 0
        1 | 1 | 0
        """
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Variable(eta(self.l2.x - self.l1.x))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(IMPLICATION_REGEX, string)


class ReverseImplication(Function2):

    priority: int = REVERSE_IMPLICATION_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        """
        Logic implication between two variable (x -> y)

        x | y | r
        0 | 0 | 0
        0 | 1 | 0
        1 | 0 | 1
        1 | 1 | 0
        """
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Variable(eta(self.l1.x - self.l2.x))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(REVERSE_IMPLICATION_REGEX, string)


class DoubleImplication(Function2):

    priority: int = DOUBLE_IMPLICATION_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        """
        Logic implication between two variable (x <-> y)

        x | y | r
        0 | 0 | 0
        0 | 1 | 1
        1 | 0 | 1
        1 | 1 | 0
        """
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Variable(eta(tf.abs(self.l1.x - self.l2.x)))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(DOUBLE_IMPLICATION_REGEX, string)


class Negation(Function1):

    priority: int = NEGATION_PRIORITY

    def __init__(self, v: Variable):
        """
        Logic negation of a variable.
        """
        super().__init__(v)

    def compute(self) -> Variable:
        return Variable(Variable.false() - eta(self.v.x))

    def copy(self) -> LogicElement:
        return Negation(self.v)

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(NEGATION_REGEX, string)


class Conjunction(Function2):

    priority: int = CONJUNCTION_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        """
        Logic conjunction between two variables (x ^ y).
        """
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Variable(tf.maximum(self.l1.x, self.l2.x))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(CONJUNCTION_REGEX, string)


class Disjunction(Function2):

    priority: int = DISJUNCTION_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        """
        Logic disjunction between two variables (x ∨ y, this is \vee not v!).
        """
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Variable(tf.minimum(self.l1.x, self.l2.x))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(DISJUNCTION_REGEX, string)


class GreaterEqual(Function2):

    priority: int = GREATER_EQUAL_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Variable(eta(Variable.false() - eta(self.l1.x - self.l2.x)))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(GREATER_EQUAL_REGEX, string)


class Greater(Function2):

    priority: int = GREATER_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Conjunction(GreaterEqual(self.l1, self.l2).compute(), NotEqual(self.l1, self.l2).compute()).compute()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(GREATER_REGEX, string)


class NotEqual(Function2):

    priority: int = NOT_EQUAL_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Negation(Equivalence(self.l1, self.l2).compute()).compute()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(NOT_EQUAL_REGEX, string)


class Less(Function2):

    priority: int = LESS_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Negation(GreaterEqual(self.l1, self.l2).compute()).compute()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(LESS_REGEX, string)


class LessEqual(Function2):

    priority: int = LESS_EQUAL_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Negation(Greater(self.l1, self.l2).compute()).compute()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(LESS_EQUAL_REGEX, string)


class Plus(Function2):

    priority: int = PLUS_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Variable(self.l1.value + self.l2.value)

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(PLUS_REGEX, string)


class Product(Function2):

    priority: int = PRODUCT_PRIORITY

    def __init__(self, l1: Variable, l2: Variable):
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        return Variable(self.l1.value * self.l2.value)

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(PRODUCT_REGEX, string)


class Output(Variable):

    priority: int = CLASS_PRIORITY

    def __init__(self, x: Tensor):
        """
        :param x: is a 1D tensorflow tensor
        """
        super().__init__(x)

    def compute(self) -> Output:
        return self


class OutputVariable(Output):

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(CLASS_X_REGEX, string)


class OutputConstant(Output):

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(CLASS_Y_REGEX, string)


class OutputEquivalence(Function2):

    priority: int = EQUIVALENCE_PRIORITY

    def __init__(self, l1: Output, l2: Output):
        """
        Element wise logic equivalence between the two tensors.
        """
        super().__init__(l1, l2)

    def compute(self) -> Variable:
        """
        :return: the maximum element wise equivalence value between the two tensors.
        """
        xy = tf.stack([self.l1.x, tf.tile(tf.reshape(self.l2.x, [1, self.l2.x.shape[0]]),
                                          [tf.shape(self.l1.x)[0], 1])], axis=1)
        element_wise_equivalence = tf.map_fn(lambda x: Equivalence(Variable(x[0, :]),
                                                                   Variable(x[1, :])).compute().value, xy)
        result = tf.reduce_max(element_wise_equivalence, axis=1)
        return Variable(eta(result))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicElement._parse(LT_EQUIVALENCE_REGEX, string)
