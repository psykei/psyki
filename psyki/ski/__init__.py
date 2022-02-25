from abc import ABC, abstractmethod
from typing import Callable, Any, Iterable
from tensorflow import Tensor, stack
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Concatenate, Lambda
from tensorflow.python.keras.saving.save import load_model
from psyki.datalog import Fuzzifier
from resources.dist.resources.DatalogParser import DatalogParser


class Injector(ABC):
    """
    An injector is a class that allows a sub-symbolic predictor to exploit prior symbolic knowledge.
    The knowledge is provided via rules in some sort of logic form (e.g. FOL, Skolem, Horn).
    """
    predictor: Any  # Any class that has methods fit and predict

    @abstractmethod
    def inject(self, rules: dict[str, DatalogParser]) -> None:
        pass


class ConstrainingInjector(Injector):

    def __init__(self, predictor: Model, class_mapping: dict[str, int],
                 feature_mapping: dict[str, int], gamma: float = 1.):
        self.predictor: Model = predictor
        self.class_mapping: dict[str, int] = class_mapping
        self.feature_mapping: dict[str, int] = feature_mapping
        self.gamma: float = gamma
        self._fuzzy_functions: Iterable[Callable] = ()

    def inject(self, rules: dict[str, DatalogParser]) -> None:
        visitor = Fuzzifier(self.class_mapping, self.feature_mapping)
        for _, rule in rules.items():
            visitor.visit(rule.formula())
        self._fuzzy_functions = [visitor.classes[name] for name in sorted(self.class_mapping.keys(),
                                                                          key=lambda i: i[1])]
        predictor_output = self.predictor.layers[-1].output
        x = Concatenate(axis=1)([self.predictor.input, predictor_output])
        x = Lambda(self._cost, self.predictor.output.shape)(x)
        self.predictor = Model(self.predictor.input, x)

    def _cost(self, output_layer: Tensor) -> Tensor:
        input_len = self.predictor.input.shape[1]
        x, y = output_layer[:, :input_len], output_layer[:, input_len:]
        cost = stack([function(x, y) for function in self._fuzzy_functions], axis=1)
        return y + (cost / self.gamma)

    def remove(self) -> None:
        """
        Remove the constraining obtained by the injected rules.
        """
        self.predictor = Model(self.predictor.input, self.predictor.layers[-3].output)

    def load(self, file):
        return load_model(file, custom_objects={'_knowledge_function': self._cost})
