from tensorflow.keras import Model
from tensorflow.python.keras.models import load_model
from psyki.logic import *
from tensorflow.keras.layers import Concatenate, Lambda, Dense
from psyki.utils import eta


class Injector(ABC):

    def __init__(self, predictor, parser: Parser = Parser.default_parser()):
        self.predictor = predictor
        self.parser = parser

    def inject(self, rules: dict[str, str], activation, input_mapping: dict, output_mapping: dict = None) -> None:
        """
        Inject symbolic knowledge in form of logic rules into the predictor.
        :param rules: rules to be injected
        :param activation: activation function of the output level
        :param input_mapping: mapping between features and variables
        :param output_mapping: mapping between classes and constants
        """
        abstract_method_exception()

    @property
    def predictor_input(self) -> Tensor:
        return self.predictor.layers[0].input


class ConstrainingInjector(Injector):

    def __init__(self, predictor, parser: Parser = Parser.default_parser(), gamma: float = 1.):
        super().__init__(predictor, parser)
        self.gamma = gamma
        self._functions = None

    def inject(self, rules: dict[str, str], activation, input_mapping: dict, output_mapping: dict = None) -> None:
        self._functions = [self.parser.function(rule, input_mapping, output_mapping) for _, rule in rules.items()]
        network_output: Tensor = self.predictor.layers[-1].output
        x = Concatenate(axis=1)([self.predictor_input, network_output])
        x = Lambda(self._cost_function, self.predictor.output.shape)(x)
        self.predictor = Model(self.predictor_input, x)

    def remove(self):
        self.predictor = Model(self.predictor_input, self.predictor.layers[-3].output)

    def _cost_function(self, layer_output: Tensor) -> Tensor:
        input_len = self.predictor_input.shape[1]
        x, y = layer_output[:, :input_len], layer_output[:, input_len:]
        cost_tensor = tf.stack([expression(x, y).value for expression in self._functions], axis=1)
        result = y + (cost_tensor / self.gamma)
        return result


class StructuringInjector(Injector):

    def __init__(self, predictor, parser: Parser = Parser.default_parser()):
        super().__init__(predictor, parser)

    def inject(self, rules: dict[str, str], activation, input_mapping, output_mapping=None) -> None:
        network_output: Tensor = self.predictor.layers[-2].output
        modules = self.modules(rules, self.predictor_input, input_mapping)
        neurons: int = self.predictor.layers[-1].output.shape[1]
        new_network = Dense(neurons, activation=activation)(Concatenate(axis=1)([network_output] + list(modules)))
        self.predictor = Model(self.predictor_input, new_network)

    def modules(self, rules: dict[str, str], network_input, input_mapping) -> Iterable:
        trees = [self.parser.structure(rule, True) for _, rule in rules.items()]
        return [self.module(tree, network_input, input_mapping) for tree in trees]

    # TODO: refactor all this block to improve readability and extendability
    def module(self, tree: Node, predictor_input, input_mapping, current_node=None):
        current_node = tree if current_node is None else current_node
        if len(current_node.children) == 0:
            if current_node.logic_element == Skip:
                return current_node.logic_element.layer(predictor_input)
            elif current_node.logic_element == Constant:
                return current_node.logic_element.layer([float(current_node.arg), predictor_input])
            else:  # Filtering and Identity
                return current_node.logic_element.layer([input_mapping[current_node.arg], predictor_input])
        elif len(current_node.children) == 1:
            # Negation
            previous_layer = self.module(tree, predictor_input, input_mapping, current_node.children[0])
            return current_node.logic_element.layer(previous_layer)
        else:
            previous_layer = [self.module(tree, predictor_input, input_mapping, child) for child in current_node.children]
            if current_node.logic_element in [Conjunction, Disjunction, Product]:
                return current_node.logic_element.layer(previous_layer)
            else:
                return current_node.logic_element.layer(Concatenate(axis=1)(previous_layer))

    @staticmethod
    def load_model(file: str):
        return load_model(file, custom_objects={'eta': eta,
                                                'eta_abs': eta_abs,
                                                'eta_abs_one': eta_abs_one,
                                                'eta_one_abs': eta_one_abs})
