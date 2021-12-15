from typing import Callable
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.layers import Lambda


class Injector:

    def __init__(self, predictor):
        self.predictor = predictor
        self.use_knowledge: bool = False
        self.rules: list = []
        self.activation_function: Callable = lambda _: _

    # Todo: ponder if it is reasonable to add a constant tensor for weighting
    def inject(self, rules: list[Callable]) -> None:
        self.use_knowledge = True
        self.rules = rules
        self.activation_function = self.predictor.layers[-1].activation
        # self.predictor.layers[-1].activation = self._altered_activation_function
        self.predictor.add(Lambda(self._altered_activation_function, output_shape=self.predictor.layers[-1].output))

    def _altered_activation_function(self, layer_output) -> None:
        net_input = self.predictor.layers[0].input
        iteration_input = tf.concat((net_input, layer_output), axis=1)
        input_len = self.predictor.layers[0].input.shape[1]
        if self.use_knowledge:
            layer_output = tf.scan(self._cost_function,
                                   elems=iteration_input,
                                   initializer=tf.zeros((input_len, ), dtype=tf.float32))
            return self.activation_function(layer_output)
        else:
            return layer_output

    def _cost_function(self, accumulator, x_and_y) -> None:
        input_len = self.predictor.layers[0].input.shape[1]
        x, y = x_and_y[:input_len], x_and_y[input_len:]
        cost_tensor = tf.stack([expression(x, y).get_value() for expression in self.rules], axis=0)
        return accumulator + (y + cost_tensor)

    @property
    def knowledge(self) -> bool:
        return self.use_knowledge

    @knowledge.setter
    def knowledge(self, value: bool):
        self.use_knowledge = value
