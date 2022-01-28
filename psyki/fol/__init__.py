from __future__ import annotations
import itertools
from typing import Iterable, Any, Union, Callable
from tensorflow.python.ops.array_ops import gather
from psyki.fol.operators import *


class Parser:

    def __init__(self, accepted_operators: Iterable[LogicOperator.__class__]):
        self.accepted_operators = accepted_operators
        self.last_tree = None

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
        self.last_tree = tree.root.copy()
        return tree.root.call(input_mapping, output_mapping)

    def tree(self, rule, flat=False):
        terms = self.parse(rule)
        tree = AST()
        for term in terms:
            tree.insert(term[0], term[1])
        return tree.root.children[0].flat_tree() if flat else tree.root.children[0]

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
        return Parser([L, LTX, LTY, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar, Implication, Exist,
                       Disjunction, Plus, Negation, Numeric, Product, Disequal, DoubleImplication, LessEqual, Pass,
                       LTEquivalence])

    @staticmethod
    def default_parser():
        return Parser([L, LTX, LTY, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar, Implication,
                       Disjunction, Plus, Negation, Numeric, Product, Disequal, DoubleImplication, LessEqual, Pass,
                       LessEqual, Greater, LTEquivalence])


class AST:

    def __init__(self, parent_ast=None):
        self.root: Union[Node, None] = None
        self.tmp_ast: Union[AST, None] = None
        self.parent_ast: Union[AST, None] = parent_ast

    def __str__(self) -> str:
        return self.root.__str__()

    def insert(self, lo: LogicOperator.__class__, arg: Any) -> None:
        # If there is already an open left par call insert on the tmp ast
        if self.tmp_ast is not None:
            self.tmp_ast.insert(lo, arg)
        # If there is a right par close the tmp ast
        elif lo == RightPar and self.parent_ast is not None:
            # If the parent ast is not empty append this ast as the rightmost child of the first incomplete node
            if self.parent_ast.root is not None:
                incomplete_node = self.parent_ast.root.get_first_incomplete_node()
                if incomplete_node is None:
                    self.parent_ast.root.children.append(self.root)
                    self.root.father = self.parent_ast.root
                else:
                    incomplete_node.children.append(self.root)
                    self.root.father = incomplete_node
            else:
                self.parent_ast.root = self.root
            self.parent_ast.tmp_ast = None
        # Create a new tmp ast if there is a left par
        elif lo == LeftPar:
            self.tmp_ast = AST(self)
        # Base case: empty AST
        elif self.root is None:
            self.root = Node(lo, arg)
        # If there is a variable in the root change it with the new lo, the variable become a child
        elif self.root.operator == L or self.root.operator == LT:
            self.root = Node(lo, arg, [self.root])
        # If ast is not complete
        elif not self.root.is_complete():
            self.root.insert(lo, arg)
        # Add ol as new root if AST is complete and there is no priority
        elif self.root.is_complete() and lo.priority <= self.root.operator.priority:
            self.root = Node(lo, arg, [self.root])
        # Change rightmost child if there is priority and AST is complete
        elif self.root.is_complete() and lo.priority > self.root.operator.priority:
            rightmost_child = self.root.children[-1]
            new_node = Node(lo, arg, [rightmost_child], self.root)
            rightmost_child.father = new_node
            self.root.children[-1] = new_node


class Node:

    def __init__(self, lo: LogicOperator.__class__, arg: Any, children: list = None, father: Node = None):
        self.operator: LogicOperator.__class__ = lo
        self.arg = arg
        self.father = father
        self.children: list[Node] = [] if children is None else children

    def __str__(self, level: int = 0):
        result = "\t" * level + repr(self.arg) + '\n'
        return result + "".join(child.__str__(level + 1) for child in self.children)

    def __getitem__(self, item, count=0):
        if item == count:
            return self
        else:
            for child in self.children:
                count = count + 1
                return child[item, count]

    def copy(self) -> Node:
        return Node(self.operator, self.arg, self.children, self.father)

    @property
    def depth(self) -> int:
        if self.father is None:
            return 0
        else:
            return 1 + self.father.depth

    def flat_tree(self, father=None) -> Node:
        if len(self.children) > 0:
            flat_operator = self.operator
            # flat_operator.arity = self._equal_operator_depth()
            new_node = Node(flat_operator, self.arg, father=father)
            new_children = [node.flat_tree(new_node) for node in self._fringe()]
            new_node.children = new_children
            return new_node
        else:
            new_node = self.copy()
            new_node.father = father
            return new_node

    def call(self, input_mapping: dict, output_mapping: dict = None) -> Callable:
        im, om = input_mapping, output_mapping
        if self.operator == Pass:
            return lambda x, y: self.operator(L(x), LTX(y)).compute()
        elif self.operator == LTX:
            return lambda x: self.operator(x).compute()
        elif self.operator == LTY:
            return lambda _: self.operator(output_mapping[self.arg]).compute()
        elif self.operator == Numeric:
            return lambda _: self.operator(self.arg).compute()
        elif self.operator == L:
            return lambda x: self.operator(x[:, input_mapping[self.arg]]).compute()
        elif self.operator.arity == 0:
            if self.operator == Exist:
                return Node._exist(self.operator, self.arg, im)
        elif self.operator.arity == 1:
            return lambda x: self.operator(self.children[0].call(im, om)(x)).compute()
        elif self.operator.arity == 2 and (self.operator == Implication or self.operator == DoubleImplication or
                                           self.operator == ReverseImplication):
            return lambda x, y: self.operator(self.children[0].call(im, om)(x),
                                              self.children[1].call(im, om)(y)).compute()
        else:
            return lambda x: self.operator(self.children[0].call(im, om)(x), self.children[1].call(im, om)(x)).compute()

    def is_complete(self) -> bool:
        if self.operator == L or self.operator == LT:
            return True
        elif self.operator.arity == len(self.children):
            return all([node.is_complete() for node in self.children])
        else:
            return False

    # TODO: father is not working as expected
    def insert(self, lo: LogicOperator.__class__, arg: Any):
        done: bool = False
        for node in self.children:
            if done:
                break
            done = node._insert(lo, arg, done)
        if not done:
            self.children.append(Node(lo, arg, father=self))

    def get_first_incomplete_node(self) -> Union[None, Node]:
        if self.operator.arity == 0:
            return None
        elif len(self.children) > 0:
            children_results = [node.get_first_incomplete_node() for node in self.children]
            for result in children_results:
                if result is not None:
                    return result
            return None if len(self.children) == self.operator.arity else self
        else:
            return self

    def _insert(self, lo: LogicOperator.__class__, arg: Any, done: bool) -> bool:
        for node in self.children:
            if done:
                return True
            done = node._insert(lo, arg, done)

        if not done and len(self.children) < self.operator.arity:
            self.children.append(Node(lo, arg, father=self))
            return True
        elif not done:
            return False

    @staticmethod
    def _exist(operator: Exist.__class__, arg: Any, input_mapping: dict) -> Callable:
        local_vars = arg[0]
        expression = arg[1]
        vars = arg[2]
        ast = AST()
        for op, local_arg in expression:
            ast.insert(op, local_arg)
        indices = [input_mapping.get(name) for name in vars]
        mapping = {name: index for index, name in enumerate(vars)}
        return lambda x: operator(local_vars, mapping, ast, gather(x, indices, axis=1)).compute()

    def _equal_operator_depth(self, result=0) -> int:
        if self.operator == self.children[0].operator:
            if len(self.children) == 2 and self.operator == self.children[1].operator:
                return result + self.children[0]._equal_operator_depth(result) + self.children[1]._equal_operator_depth(result) + 1
            else:
                return result + self.children[0]._equal_operator_depth(result) + 1
        else:
            return result + self.operator.arity

    def leaves(self) -> Iterable[Node]:
        if len(self.children) == 0:
            return self,
        else:
            return list(itertools.chain.from_iterable(child.leaves() for child in self.children))

    def level(self, l=0) -> Iterable[Node]:
        if self.depth == l:
            return self,
        else:
            return list(itertools.chain.from_iterable(child.level(l) for child in self.children))

    def prune_leaves(self):
        if len(self.children) > 0:
            children_copy = self.children.copy()
            for node in children_copy:
                node.prune_leaves()
        else:
            self.father.children.remove(self)

    def prune_constants(self):
        if len(self.children) > 0:
            children_copy = self.children.copy()
            for node in children_copy:
                node.prune_constants()
        else:
            if self.operator == Numeric:
                self.father.children.remove(self)

    def _fringe(self) -> list[Node]:
        if self.operator == self.children[0].operator:
            if len(self.children) == 2 and self.operator == self.children[1].operator:
                return self.children[0]._fringe() + self.children[1]._fringe()
            else:
                return self.children[0]._fringe() + [self.children[1]] if len(self.children) == 2 else self.children[0]._fringe()
        else:
            return self.children
