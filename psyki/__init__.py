from typing import Callable
from collections import Iterable
from tensorflow.keras import Model
from tensorflow.python.keras.backend import shape, constant, relu
from tensorflow.python.keras.layers import Minimum, Maximum, Dot
from tensorflow.python.ops.array_ops import gather
from tensorflow.python.ops.init_ops_v2 import constant_initializer, Ones, Zeros
import tensorflow as tf
from psyki.fol import Node, Conjunction, Disjunction, Equivalence, Disequal, GreaterEqual, Greater, Less, LessEqual, \
    Plus, Product, Numeric, Pass
from tensorflow.keras.layers import Concatenate, Lambda, Input, Dense
from tensorflow.keras.models import load_model
from tensorflow import Tensor, stack


class Injector:

    def __init__(self, predictor, input, output_shape: int = 10, gamma: float = 1.):
        self.original_predictor = predictor
        self.predictor = predictor
        self.input = input
        self.use_knowledge: bool = False
        self.rules: list = []
        self.active_rule: list = []
        self.gamma = gamma
        self.output = output_shape

    def inject(self, rules: list[Callable], active_rule: list[Callable] = None) -> None:
        self.use_knowledge = True
        self.rules = rules
        self.active_rule = active_rule
        x = Concatenate(axis=1, name='Concatenate')([self.input, self.original_predictor])
        x = Lambda(self._knowledge_function, self.output, name='Knowledge')(x)
        self.predictor = Model(self.input, x)

    def _knowledge_function(self, layer_output: Tensor) -> Tensor:
        output_len = self.original_predictor.shape[1]
        if self.use_knowledge:
            return self._cost_function(layer_output)
        else:
            return layer_output[:, -output_len:]

    def _cost_function(self, x_and_y: Tensor) -> Tensor:
        input_len = self.input.shape[1]
        x, y = x_and_y[:, :input_len], x_and_y[:, input_len:]
        cost_tensor = stack([expression(x, y).get_value() for expression in self.rules], axis=1)
        result = y + (cost_tensor / self.gamma)
        return result

    def save(self, file: str):
        Model(inputs=self.predictor.input, outputs=self.predictor.layers[-3].output).save(file)

    def load(self, file):
        return load_model(file, custom_objects={'_knowledge_function': self._knowledge_function})

    @property
    def knowledge(self) -> bool:
        return self.use_knowledge

    @knowledge.setter
    def knowledge(self, value: bool):
        self.use_knowledge = value


class KnowledgeModule:

    def __init__(self, tree: Node, network_input: Input, input_mapping):
        self.tree = tree
        self.input = network_input
        self.input_mapping = input_mapping
        # self.tree.prune_constants()
        self.activation = 'tanh'

    def network(self, current_node=None):
        current_node = self.tree if current_node is None else current_node
        if len(current_node.children) == 0:
            index = self.input_mapping[current_node.arg]
            return Lambda(lambda x: gather(x, [index], axis=1))(self.input)
        elif len(current_node.children) == 1:
            return Dense(1, activation=self.activation)(self.network(current_node=current_node.children[0]))
        else:
            previous_layer = Concatenate(axis=1)([self.network(current_node=child) for child in current_node.children])
            return Dense(1, activation=self.activation)(previous_layer)

    def initialized_network(self, current_node=None):
        current_node = self.tree if current_node is None else current_node
        if len(current_node.children) == 0:
            if current_node.operator == Pass:
                return Dense(1, kernel_initializer=Zeros, bias_initializer=constant_initializer(0))(self.input)
            elif current_node.operator == Numeric:
                return Dense(1, kernel_initializer=Zeros, bias_initializer=constant_initializer(float(current_node.arg)),
                             trainable=False, activation='linear')(self.input)
            else:  # Filtering and Identity
                index = self.input_mapping[current_node.arg]
                return Lambda(lambda x: gather(x, [index], axis=1))(self.input)
        elif len(current_node.children) == 1:
            # Negation
            return Dense(1, kernel_initializer=Ones, activation=KnowledgeModule.negation, trainable=False)\
                (self.initialized_network(current_node=current_node.children[0]))
        else:
            previous_layer = Concatenate(axis=1)\
                ([self.initialized_network(current_node=child) for child in current_node.children])
            if current_node.operator == Conjunction:
                return Minimum()([self.initialized_network(current_node=child) for child in current_node.children])
            elif current_node.operator == Disjunction:
                return Maximum()([self.initialized_network(current_node=child) for child in current_node.children])
            elif current_node.operator == Equivalence:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]), activation=KnowledgeModule.one_minus_abs, trainable=False)(previous_layer)
            elif current_node.operator == Disequal:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]), activation=KnowledgeModule.my_abs, trainable=False)\
                    (previous_layer)
            elif current_node.operator == Greater:
                return Dense(1, kernel_initializer=constant_initializer([1, -1]), activation='relu', trainable=False)(previous_layer)
            elif current_node.operator == Less:
                return Dense(1, kernel_initializer=constant_initializer([-1, 1]), activation='relu', trainable=False)(previous_layer)
            elif current_node.operator == GreaterEqual:
                greater = Dense(1, kernel_initializer=constant_initializer([1, -1]), activation='relu', trainable=False)(previous_layer)
                equal = Dense(1, kernel_initializer=constant_initializer([1, -1]), activation=KnowledgeModule.one_minus_abs, trainable=False)\
                    (previous_layer)
                return Maximum()([greater, equal])
            elif current_node.operator == LessEqual:
                less = Dense(1, kernel_initializer=constant_initializer([-1, 1]), activation='relu', trainable=False)(previous_layer)
                equal = Dense(1, kernel_initializer=constant_initializer([1, -1]), activation=KnowledgeModule.one_minus_abs, trainable=False)\
                    (previous_layer)
                return Maximum()([less, equal])
            elif current_node.operator == Plus:
                return Dense(1, kernel_initializer=Ones(), activation='linear', trainable=False)(previous_layer)
            elif current_node.operator == Product:
                return Dot(axes=1)([self.initialized_network(current_node=child) for child in current_node.children])
            else:
                return Dense(1, activation=self.activation, trainable=False)(previous_layer)

    @staticmethod
    def my_abs(x):
        return tf.minimum(1, tf.abs(x))

    @staticmethod
    def one_minus_abs(x):
        return relu(1 - tf.abs(x))

    @staticmethod
    def negation(x):
        return tf.abs(x - 1)
