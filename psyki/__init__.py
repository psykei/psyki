from tensorflow.keras import Model
from tensorflow.python.keras.layers import Minimum, Maximum, Dot
from tensorflow.python.keras.models import load_model
from tensorflow.python.ops.array_ops import gather
from tensorflow.python.ops.init_ops_v2 import constant_initializer, Ones, Zeros
from psyki.logic import *
from tensorflow.keras.layers import Concatenate, Lambda, Dense
from psyki.utils import eta


class Injector:

    def __init__(self, predictor, parser: Parser = Parser.default_parser()):
        self.predictor = predictor
        self.parser = parser

    def inject(self, rules: dict[str, str], activation, input_mapping: dict, output_mapping: dict = None):
        abstract_method_exception()


class StructuringInjector(Injector):

    def __init__(self, predictor, parser: Parser = Parser.default_parser()):
        super().__init__(predictor, parser)

    def inject(self, rules: dict[str, str], activation, input_mapping, output_mapping=None):
        network_input: Tensor = self.predictor.input
        network_output: Tensor = self.predictor.layers[-2].output
        neurons: int = self.predictor.layers[-1].output
        modules = self.modules(rules, network_input, input_mapping)
        return Model(network_input,
                     Dense(neurons, activation=activation)(Concatenate(axis=1)([network_output] + list(modules))))

    def modules(self, rules: dict[str, str], network_input, input_mapping) -> Iterable:
        trees = [self.parser.structure(rule, True) for _, rule in rules.items()]
        return [self.module(tree, network_input, input_mapping) for tree in trees]

    def module(self, tree, network_input, input_mapping, current_node=None):
        current_node = tree if current_node is None else current_node
        if len(current_node.children) == 0:
            if current_node.logic_element == Skip:
                return Dense(1, kernel_initializer=Zeros, bias_initializer=constant_initializer(0))(network_input)
            elif current_node.logic_element == Constant:
                return Dense(1, kernel_initializer=Zeros,
                             bias_initializer=constant_initializer(float(current_node.arg)),
                             trainable=False, activation='linear')(network_input)
            else:  # Filtering and Identity
                index = input_mapping[current_node.arg]
                return Lambda(lambda x: gather(x, [index], axis=1))(network_input)
        elif len(current_node.children) == 1:
            # Negation
            return Dense(1, kernel_initializer=Ones, activation=StructuringInjector.negation, trainable=False) \
                (self.module(tree, network_input, input_mapping, current_node.children[0]))
        else:
            # TODO: refactor all this block to improve readability and extendability
            previous_layer = Concatenate(axis=1) \
                ([self.module(tree, network_input, input_mapping, child) for child in current_node.children])
            if current_node.logic_element == Conjunction:
                return Minimum()(
                    [self.module(tree, network_input, input_mapping, child) for child in current_node.children])
            elif current_node.logic_element == Disjunction:
                return Maximum()(
                    [self.module(tree, network_input, input_mapping, child) for child in current_node.children])
            elif current_node.logic_element == Equivalence:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]),
                             activation=StructuringInjector.one_minus_abs, trainable=False)(previous_layer)
            elif current_node.logic_element == NotEqual:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]), activation=StructuringInjector.my_abs,
                             trainable=False) \
                    (previous_layer)
            elif current_node.logic_element == Greater:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]), activation='relu', trainable=False)(
                    previous_layer)
            elif current_node.logic_element == Less:
                return Dense(1, kernel_initializer=constant_initializer([-1, 1]), activation='relu', trainable=False)(
                    previous_layer)
            elif current_node.logic_element == GreaterEqual:
                greater = Dense(1, kernel_initializer=constant_initializer([1, -1]), activation='relu',
                                trainable=False)(previous_layer)
                equal = Dense(1, kernel_initializer=constant_initializer([1, -1]),
                              activation=StructuringInjector.one_minus_abs, trainable=False) \
                    (previous_layer)
                return Maximum()([greater, equal])
            elif current_node.logic_element == LessEqual:
                less = Dense(1, kernel_initializer=constant_initializer([-1, 1]), activation='relu', trainable=False)(
                    previous_layer)
                equal = Dense(1, kernel_initializer=constant_initializer([1, -1]),
                              activation=StructuringInjector.one_minus_abs, trainable=False) \
                    (previous_layer)
                return Maximum()([less, equal])
            elif current_node.logic_element == Plus:
                return Dense(1, kernel_initializer=Ones(), activation='linear', trainable=False)(previous_layer)
            elif current_node.logic_element == Product:
                return Dot(axes=1)(
                    [self.module(tree, network_input, input_mapping, child) for child in current_node.children])
            else:
                return Dense(1, activation=eta, trainable=False)(previous_layer)

    @staticmethod
    def load_model(file: str):
        return load_model(file, custom_objects={'my_abs': StructuringInjector.my_abs,
                                                'one_minus_abs': StructuringInjector.one_minus_abs,
                                                'negation': StructuringInjector.negation,
                                                'eta': eta})

    @staticmethod
    def my_abs(x):
        return eta(tf.abs(x))

    @staticmethod
    def one_minus_abs(x):
        return eta(1 - tf.abs(x))

    @staticmethod
    def negation(x):
        return eta(tf.abs(x - 1))
