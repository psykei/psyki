from typing import Iterable, Any
from psyki.fol.ast import AST
from psyki.fol.operators import *


class Parser:

    def __init__(self, accepted_operators: Iterable[LogicOperator.__class__]):
        self.accepted_operators = accepted_operators

    def parse(self, string: str) -> list[tuple[LogicOperator.__class__, Any]]:
        results = []
        string = string.replace(' ', '')
        while len(string) > 0:
            old_string = string
            for op in self.accepted_operators:
                match, value = op.parse(string)
                if match:
                    string = string.replace(value, '', 1)
                    if op == Exist:
                        value = self._get_exist_value(value)
                    results.append((op, value))
                    break
            if old_string == string:
                raise Exception('Parser cannot parse the provided string: ' + string)
        return results

    def get_function(self, rule, input_mapping, output_mapping):
        terms = self.parse(rule)
        tree = AST()
        for term in terms:
            tree.insert(term[0], term[1])
        return tree.root.call(input_mapping, output_mapping)

    def _get_exist_value(self, string: str) -> Any:
        # ∃(local vars: expression, vars)
        _, string = Exist.parse_head(string)
        local_vars, string = Exist.parse_local_vars(string)
        local_vars = [var for var in local_vars.split(',')]
        expression, string = Exist.parse_expression(string)
        expression = self.parse(expression)
        vars, string = Exist.parse_vars(string)
        vars = [var for var in vars.split(',')]
        return local_vars, expression, vars

    @staticmethod
    def extended_parser():
        return Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar,
                       Implication, Exist, Disjunction, Plus, Negation, Numeric, Product, Disequal, DoubleImplication,
                       LessEqual, Pass])

    @staticmethod
    def default_parser():
        return Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar,
                       Implication, Disjunction, Plus, Negation, Numeric, Product, Disequal, DoubleImplication,
                       LessEqual, Pass, LessEqual, Greater])
