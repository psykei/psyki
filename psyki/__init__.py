from typing import Callable
from collections import Iterable
from tensorflow.keras import Model
from tensorflow.python.ops.array_ops import gather
from psyki.fol import Node
from tensorflow.keras.layers import Concatenate, Lambda, Input, Dense
from tensorflow.keras.models import load_model
from tensorflow import Tensor, stack


class Injector:

    def __init__(self, predictor, input, output_shape: int = 10, activation_function: Callable = None,
                 gamma: float = 1.):
        self.original_predictor = predictor
        self.predictor = predictor
        self.input = input
        self.use_knowledge: bool = False
        self.rules: list = []
        self.active_rule: list = []
        self.activation_function = activation_function
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
        Model(inputs=self.predictor.net_input, outputs=self.predictor.layers[-3].output).save(file)

    def load(self, file):
        return load_model(file, custom_objects={'_knowledge_function': self._knowledge_function})

    @property
    def knowledge(self) -> bool:
        return self.use_knowledge

    @knowledge.setter
    def knowledge(self, value: bool):
        self.use_knowledge = value


class KnowledgeNetwork:

    def __init__(self, tree: Node):
        self.tree = tree
        self.input = None

    def network(self):
        variables = self.tree.leaves()
        input_size = KnowledgeNetwork._unique_nodes(variables)
        unique_names = []
        # Preserve order
        for variable in variables:
            if variable.arg not in unique_names:
                unique_names.append(variable.arg)
        variables_mapping = {name: i for name, i in zip(unique_names, range(input_size))}
        self.input = Input(input_size)
        connection_mapping = self._connections(variables_mapping)
        self.tree.prune_leaves()
        new_layer = self.build_layer(self.input, connection_mapping)
        while len(self.tree.children) > 0:
            previous_layer = new_layer
            old_nodes = list(self.tree.leaves())
            previous_layer_size = len(old_nodes)
            for i, node in enumerate(old_nodes):
                node.arg = i
            naming = {node.arg: i for node, i in zip(old_nodes, range(previous_layer_size))}
            connection_mapping = self._connections(naming)
            self.tree.prune_leaves()
            if len(self.tree.children) > 0:
                new_layer = self.build_layer(previous_layer, connection_mapping)
        return Dense(1)(new_layer)

    def build_layer(self, previous_layer, mapping):
        layer_size = len(list(self.tree.leaves()))
        inputs = [Lambda(lambda x: gather(x, mapping[i], axis=1), output_shape=(len(mapping[i]),))(previous_layer)
                  for i in range(layer_size)]
        return Concatenate()([Dense(1)(inputs[i]) for i in range(layer_size)])

    @staticmethod
    def _unique_nodes(nodes: Iterable[Node]) -> int:
        return len(set([node.arg for node in nodes]))

    def _connections(self, naming) -> dict[int, list[int]]:
        leaves_depth = list(self.tree.leaves())[0].depth
        new_nodes = self.tree.level(leaves_depth - 1)
        result = {}
        for i, node in enumerate(new_nodes):
            result[i] = [naming[child.arg] for child in node.children]
        return result  # {i: [naming[child.arg] for child in node.children] for i, node in enumerate(new_nodes)}
