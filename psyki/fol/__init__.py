from typing import Iterable, Any
from psyki.fol.operators import Exist, LogicOperator


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
                raise Exception('Parser cannot parse the provided string.')
        return results

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
