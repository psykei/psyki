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
        self.activation_function = activation_function

    # Todo: ponder if it is reasonable to add a constant tensor for weighting
    def inject(self, rules: list[Callable]) -> None:
        self.use_knowledge = True
        self.rules = rules
        # self.original_predictor.layers[-1].activation = self._knowledge_function
        x = Concatenate(axis=1)([self.input, self.original_predictor])
        x = Lambda(self._knowledge_function, (10,))(x)
        self.predictor = Model(self.input, x)

    def _knowledge_function(self, layer_output: Tensor) -> Tensor:
        output_len = self.original_predictor.shape[1]
        if self.use_knowledge:
            # Todo: tf.scan is devil, to avoid at any cost
            """result = tf.scan(self._cost_function,
                             elems=layer_output,
                             initializer=tf.zeros((output_len, ), dtype=tf.float32))"""
            # tensors = tf.unstack(layer_output)
            result = self._cost_function(layer_output)
        else:
            result = layer_output[-output_len:]
        return self.activation_function(result)

    def _cost_function(self, x_and_y: Tensor) -> Tensor:
        input_len = self.input.shape[1]
        x, y = x_and_y[:, :input_len], x_and_y[:, input_len:]
        cost_tensor = tf.stack([expression(x, y).get_value() for expression in self.rules], axis=1)
        return y + cost_tensor

    @property
    def knowledge(self) -> bool:
        return self.use_knowledge

    @knowledge.setter
    def knowledge(self, value: bool):
        self.use_knowledge = value
