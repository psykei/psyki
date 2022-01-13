from typing import Callable
import keras
import tensorflow as tf
from keras import Model
from keras.layers import Concatenate, Lambda
from tensorflow import Tensor


class Injector:

    def __init__(self, predictor, input, activation_function: Callable = None, gamma: float = 1.):
        self.original_predictor = predictor
        self.predictor = predictor
        self.input = input
        self.use_knowledge: bool = False
        self.rules: list = []
        self.active_rule: list = []
        self.activation_function = activation_function
        self.gamma = gamma

    def inject(self, rules: list[Callable], active_rule: list[Callable] = None) -> None:
        self.use_knowledge = True
        self.rules = rules
        self.active_rule = active_rule
        x = Concatenate(axis=1, name='Concatenate')([self.input, self.original_predictor])
        x = Lambda(self._knowledge_function, (10,), name='Knowledge')(x)
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
        result = y + (cost_tensor/self.gamma)
        return result

    def save(self, file: str):
        Model(inputs=self.predictor.net_input, outputs=self.predictor.layers[-3].output).save(file)

    def load(self, file):
        return keras.models.load_model(file, custom_objects={'_knowledge_function': self._knowledge_function})

    @property
    def knowledge(self) -> bool:
        return self.use_knowledge

    @knowledge.setter
    def knowledge(self, value: bool):
        self.use_knowledge = value
