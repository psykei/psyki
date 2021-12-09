from typing import Iterable
from psyki.logic import LogicOperator


class Parser:

    def __init__(self, accepted_operators: Iterable[LogicOperator.__class__]):
        self.accepted_operators = accepted_operators

    def parse(self, string: str) -> list[tuple[LogicOperator.__class__, any]]:
        results = []
        string = string.replace(' ', '')
        while len(string) > 0:
            for op in self.accepted_operators:
                match, value = op.parse(string)
                if match:
                    results.append((op, value))
                    string = string.replace(value, '', 1)
                    break
        return results


class Visitor:

    pass