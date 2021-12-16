from __future__ import annotations
import re
from itertools import combinations
from tensorflow.python.keras.backend import constant
from tensorflow.python.types.core import Tensor
import tensorflow as tf

CLASS_PRIORITY: int = 1000
CONJUNCTION_PRIORITY: int = 100
DEFAULT_PRIORITY: int = -1
DISEQUAL_PRIORITY: int = 200
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

DEFAULT_NAME: str = 'Abstract logic operator'
CLASS_X_NAME: str = 'Class X'
CLASS_Y_NAME: str = 'Class y'
CONJUNCTION_NAME: str = 'Conjunction operator'
DISEQUAL_NAME: str = 'Disequal operator'
DISJUNCTION_NAME: str = 'Disjunction operator'
DOUBLE_IMPLICATION_NAME: str = 'Double implication operator'
EQUIVALENCE_NAME: str = 'Equivalence operator'
EXIST_NAME: str = 'Exist operator'
GREATER_EQUAL_NAME: str = 'Greater or equal operator'
GREATER_NAME: str = 'Greater operator'
IMPLICATION_NAME: str = 'Implication operator'
LESS_EQUAL_NAME: str = 'Less or equal operator'
LESS_NAME: str = 'Less operator'
LT_EQUIVALENCE_NAME: str = 'Logic tensor equivalence name'
NEGATION_NAME: str = 'Negation operator'
NUMERIC_NAME: str = 'Numeric operator'
PLUS_NAME: str = 'Plus operator'
PRODUCT_NAME: str = 'Product name'
REVERSE_IMPLICATION_NAME: str = 'Reverse implication name'
VARIABLE_NAME: str = 'Variable'

DEFAULT_REGEX: str = ''
CLASS_X_REGEX: str = 'X'
CLASS_Y_REGEX: str = '[a-z]+([A-Z]|[a-z]|[0-9])*'
CONJUNCTION_REGEX: str = r'\^'
DISEQUAL_REGEX: str = '!='
DISJUNCTION_REGEX: str = r'\∨'  # this is a descending wedge not a v! #TODO: change ∨ into something more intelligible
DOUBLE_IMPLICATION_REGEX: str = '<->'
EQUIVALENCE_REGEX: str = '='
# TODO: change ∃ into something more intelligible
EXIST_REGEX: str = r'\∃\(([A-Z]([a-z]|[A-Z])*[0-9]*,)*[A-Z]([a-z]|[A-Z])*[0-9]*\:[^,]*\,' \
                   r'([A-Z]([a-z]|[A-Z])*[0-9]*,)*[A-Z]([a-z]|[A-Z])*[0-9]*\)'  # ∃(local vars: expression, vars)
GREATER_EQUAL_REGEX: str = '>='
GREATER_REGEX: str = '>'
IMPLICATION_REGEX: str = '->'
LEFT_PAR_REGEX: str = r'\('
LESS_EQUAL_REGEX: str = '<='
LT_EQUIVALENCE_REGEX: str = r'\|='
NEGATION_REGEX: str = r'\~'
NUMERIC_REGEX: str = '[+-]?([0-9]*[.])?[0-9]+'
PLUS_REGEX: str = '[+]'
PRODUCT_REGEX: str = r'\*'
REVERSE_IMPLICATION_REGEX: str = r'<-(?!>)'
LESS_REGEX: str = r'<(?!-)'
RIGHT_PAR_REGEX: str = r'\)'
VARIABLE_REGEX: str = r'^(?!' + CLASS_X_REGEX + ')[A-Z]([a-z]|[A-Z])*[0-9]*'


class LogicOperator:
    arity: int = 0
    priority: int = DEFAULT_PRIORITY

    def __init__(self, name: str = DEFAULT_NAME):
        self.name: str = name

    def compute(self) -> LogicOperator:
        raise Exception('Try to call an abstract method')

    def get_value(self) -> Tensor:
        raise Exception('Try to call an abstract method')

    @staticmethod
    def parse(string: str):
        raise Exception('Try to call an abstract method')

    @staticmethod
    def _parse(regex: str, string: str) -> tuple[bool, str]:
        match = re.match(pattern=regex, string=string)
        return (True, string[match.span()[0]:match.span()[1]]) if match is not None and match.pos == 0 else (False, '')


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


class Op2(LogicOperator):

    arity: int = 2

    def __init__(self, l1: L, l2: L, name: str = DEFAULT_NAME):
        super().__init__(name)
        self.l1 = l1
        self.l2 = l2


class Op1(LogicOperator):

    arity: int = 1

    def __init__(self, name: str = DEFAULT_NAME):
        super().__init__(name)


class L(LogicOperator):

    priority: int = VARIABLE_PRIORITY

    def __init__(self, x: Tensor):
        """
        Logic variable, 0 is true and 1 is false.
        :param x: is a tensorflow tensor of one element (can be interpreted as a scalar)
        """

        super().__init__(VARIABLE_NAME)
        self.x = x

    def compute(self) -> L:
        return self

    def get_value(self) -> Tensor:
        return self.x

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(VARIABLE_REGEX, string)

    @staticmethod
    def relu(x: Tensor):
        return tf.maximum(x, L.true())

    @staticmethod
    def reverse_relu(x: Tensor):
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


class Numeric(L):

    priority: int = NUMERIC_PRIORITY

    def __init__(self, x: str):
        """
        Logic variable, 0 is true and 1 is false.
        :param x: is a tensorflow tensor of one element (can be interpreted as a scalar)
        """

        super().__init__(tf.constant(float(x)))
        self.name = NUMERIC_NAME

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(NUMERIC_REGEX, string)


class Exist(LogicOperator):

    priority: int = EXIST_PRIORITY
    _head_regex: str = r'\∃\('
    _local_vars_regex: str = r'([A-Z]([a-z]|[A-Z])*[0-9]*,)*[A-Z]([a-z]|[A-Z])*[0-9]*\:'
    _expression_regex: str = r'[^,;]+\,'
    _vars_regex: str = r'([A-Z]([a-z]|[A-Z])*[0-9]*,)*[A-Z]([a-z]|[A-Z])*[0-9]*\)'

    def __init__(self, local_vars: list[str], outer_mapping: dict[str, int], ast, x: Tensor):
        """
        Logic negation of a variable.
        """
        super().__init__(EXIST_NAME)
        self.inner_mapping = {name: i + len(outer_mapping.keys()) for i, name in enumerate(local_vars)}
        self.outer_mapping = outer_mapping
        self.ast = ast
        self.x = x

    def compute(self) -> L:
        mapping = self.outer_mapping | self.inner_mapping
        function = self.ast.root.call(mapping)
        result = L(L.false())
        combs = combinations(range(0, self.x.shape[1]), len(self.inner_mapping.keys()))
        for combination_indices in combs:
            result = Disjunction(result,
                                 function(tf.concat([self.x, tf.gather(self.x, combination_indices, axis=1)], axis=1))
                                 ).compute()
        return result

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(EXIST_REGEX, string)

    @staticmethod
    def parse_head(string: str) -> tuple[str, str]:
        match, value = LogicOperator._parse(Exist._head_regex, string)
        return value, string.replace(value, '', 1)

    @staticmethod
    def parse_local_vars(string: str) -> tuple[str, str]:
        match, value = LogicOperator._parse(Exist._local_vars_regex, string)
        return value[:-1], string.replace(value, '', 1)  # remove ':'

    @staticmethod
    def parse_expression(string: str) -> tuple[str, str]:
        match, value = LogicOperator._parse(Exist._expression_regex, string)
        return value[:-1], string.replace(value, '', 1)  # remove ','

    @staticmethod
    def parse_vars(string: str) -> tuple[str, str]:
        match, value = LogicOperator._parse(Exist._vars_regex, string)
        return value[:-1], string.replace(value, '', 1)  # remove ')'


class Equivalence(Op2):

    priority: int = EQUIVALENCE_PRIORITY

    def __init__(self, l1: L, l2: L):
        """
        Logic equivalence between two variables (x = y).
        """
        super().__init__(l1, l2, EQUIVALENCE_NAME)

    def compute(self) -> L:
        return L(L.fringe(tf.abs(self.l1.x - self.l2.x)))

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

    def compute(self) -> L:
        return L(L.relu(self.l2.x - self.l1.x))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(IMPLICATION_REGEX, string)


class ReverseImplication(Op2):

    priority: int = REVERSE_IMPLICATION_PRIORITY

    def __init__(self, l1: L, l2: L):
        """
        Logic implication between two variable (x -> y)

        x | y | r
        0 | 0 | 0
        0 | 1 | 0
        1 | 0 | 1
        1 | 1 | 0
        """
        super().__init__(l1, l2, REVERSE_IMPLICATION_NAME)

    def compute(self) -> L:
        return L(L.relu(self.l1.x - self.l2.x))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(REVERSE_IMPLICATION_REGEX, string)


class DoubleImplication(Op2):

    priority: int = DOUBLE_IMPLICATION_PRIORITY

    def __init__(self, l1: L, l2: L):
        """
        Logic implication between two variable (x <-> y)

        x | y | r
        0 | 0 | 0
        0 | 1 | 1
        1 | 0 | 1
        1 | 1 | 0
        """
        super().__init__(l1, l2, DOUBLE_IMPLICATION_NAME)

    def compute(self) -> L:
        return L(L.relu(tf.abs(self.l2.x - self.l1.x)))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(DOUBLE_IMPLICATION_REGEX, string)


class Negation(Op1):

    priority: int = NEGATION_PRIORITY

    def __init__(self, l1: L):
        """
        Logic negation of a variable.
        """
        super().__init__(NEGATION_NAME)
        self.l1 = l1

    def compute(self) -> L:
        return L(L.false() - L.fringe(self.l1.x))

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

    def compute(self) -> L:
        return L(tf.maximum(self.l1.x, self.l2.x))

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

    def compute(self) -> L:
        return L(tf.minimum(self.l1.x, self.l2.x))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(DISJUNCTION_REGEX, string)


class GreaterEqual(Op2):

    priority: int = GREATER_EQUAL_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, GREATER_EQUAL_NAME)

    def compute(self) -> L:
        return L(L.relu(L.false() - L.relu(self.l1.x - self.l2.x)))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(GREATER_EQUAL_REGEX, string)


class Greater(Op2):

    priority: int = GREATER_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, GREATER_NAME)

    def compute(self) -> L:
        return Conjunction(GreaterEqual(self.l1, self.l2).compute(), Disequal(self.l1, self.l2).compute()).compute()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(GREATER_REGEX, string)


class Disequal(Op2):

    priority: int = DISEQUAL_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, DISEQUAL_NAME)

    def compute(self) -> L:
        return Negation(Equivalence(self.l1, self.l2).compute()).compute()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(DISEQUAL_REGEX, string)


class Less(Op2):

    priority: int = LESS_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, LESS_NAME)

    def compute(self) -> L:
        return Negation(GreaterEqual(self.l1, self.l2).compute()).compute()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(LESS_REGEX, string)


class LessEqual(Op2):

    priority: int = LESS_EQUAL_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, LESS_EQUAL_NAME)

    def compute(self) -> L:
        return Negation(Greater(self.l1, self.l2).compute()).compute()

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(LESS_EQUAL_REGEX, string)


class Plus(Op2):

    priority: int = PLUS_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, PLUS_NAME)

    def compute(self) -> L:
        return L(self.l1.get_value() + self.l2.get_value())

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(PLUS_REGEX, string)


class Product(Op2):

    priority: int = PRODUCT_PRIORITY

    def __init__(self, l1: L, l2: L):
        super().__init__(l1, l2, PRODUCT_NAME)

    def compute(self) -> L:
        return L(self.l1.get_value() * self.l2.get_value())

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(PRODUCT_REGEX, string)


class LT(L):

    priority: int = CLASS_PRIORITY

    def __init__(self, x: Tensor):
        """
        :param x: is a 1D tensorflow tensor
        """
        super().__init__(x)

    def compute(self) -> LT:
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

    def compute(self) -> L:
        """
        :return: 'the most false value' (the maximum) among the partial results
        """
        # xy = tf.stack([self.l1.x, self.l2.x], axis=1)
        xy = tf.stack([self.l1.x,
                       tf.tile(tf.reshape(self.l2.x, [1, self.l2.x.shape[0]]), [tf.shape(self.l1.x)[0], 1])], axis=1)
        element_wise_equivalence = tf.map_fn(lambda x: Equivalence(L(x[0, :]), L(x[1, :])).compute().get_value(), xy)
        return L(L.fringe(tf.reduce_max(element_wise_equivalence)))

    @staticmethod
    def parse(string: str) -> tuple[bool, str]:
        return LogicOperator._parse(LT_EQUIVALENCE_REGEX, string)
