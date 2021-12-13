from typing import Callable
from tensorflow import Tensor
import tensorflow as tf


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
        self.predictor.layers[-1].activation = self._altered_activation_function

    def _altered_activation_function(self, layer_output: Tensor) -> Tensor:
        net_input = self.predictor.layers[0].input
        iteration_input = tf.concat((net_input, layer_output), axis=1)
        if self.use_knowledge:
            result = tf.map_fn(self._cost_function, iteration_input)
        else:
            result = self.activation_function(layer_output)
        return result

    def _cost_function(self, x_and_y: Tensor) -> Tensor:
        x, y = x_and_y[0], x_and_y[1]
        cost_tensor = tf.concat([expression(x, y).get_value() for expression in self.rules], axis=0)
        return self.activation_function(y + cost_tensor)

    @property
    def knowledge(self) -> bool:
        return self.use_knowledge

    @knowledge.setter
    def knowledge(self, value: bool):
        self.use_knowledge = value
