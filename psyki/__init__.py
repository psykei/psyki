from typing import Callable
import tensorflow as tf
from keras import Model
from keras.layers import Concatenate, Lambda
from tensorflow import Tensor


class Injector:

    def __init__(self, predictor, input, activation_function: Callable):
        self.original_predictor = predictor
        self.predictor = predictor
        self.input = input
        self.use_knowledge: bool = False
        self.rules: list = []
        self.active_rule: list = []
        self.activation_function = activation_function

    # Todo: ponder if it is reasonable to add a constant tensor for weighting
    def inject(self, rules: list[Callable], active_rule: list[Callable] = None) -> None:
        self.use_knowledge = True
        self.rules = rules
        self.active_rule = active_rule
        # self.original_predictor.layers[-1].activation = self._knowledge_function
        x = Concatenate(axis=1, name='Concatenate layer')([self.input, self.original_predictor])
        x = Lambda(self._knowledge_function, (10,), name='Knowledge layer')(x)
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
        cost_tensor = tf.stack([expression(x, y).get_value() for expression in self.rules], axis=1)

        # active = tf.stack([expression(x).get_value() for expression in self.active_rule], axis=1)
        # only_one_active = Injector._process_active_rule(active)

        return y + cost_tensor/y.shape[1]  # * only_one_active)

    # Not used
    @staticmethod
    def _process_active_rule(tensor: Tensor) -> Tensor:
        zero_to_two = tf.where(tf.equal(tensor, 0.0), 2.0, tensor)
        one_to_zero = tf.where(tf.equal(zero_to_two, 1.0), 0.0, zero_to_two)
        give_priority = one_to_zero * tf.cast(tf.range(0, one_to_zero.shape[1]), dtype=tf.float32)
        indices_axis_1 = tf.cast(tf.argmax(give_priority, axis=1), dtype=tf.int32)
        indices = tf.stack([tf.cast(tf.range(0, tf.shape(tensor)[0]), dtype=tf.int32), indices_axis_1], axis=1)
        ones = tf.where(tf.not_equal(tensor, 1.0), 1.0, tensor)
        zeros = tf.zeros(tf.shape(tensor)[0])
        return tf.tensor_scatter_nd_update(ones, indices, zeros)

    @property
    def knowledge(self) -> bool:
        return self.use_knowledge

    @knowledge.setter
    def knowledge(self, value: bool):
        self.use_knowledge = value
