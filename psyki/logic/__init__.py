from __future__ import annotations
import itertools
from typing import Iterable, Any, Union, Callable
from psyki.logic.elements import *
from psyki.utils import parse_exception


class Parser:

    def __init__(self, accepted_elements: Iterable[LogicElement.__class__]):
        self.accepted_elements = accepted_elements

    def function(self, rule: str, input_mapping: dict, output_mapping: dict) -> Callable:
        """
        Return a fuzzy function that can be callable by passing the required input
        :param rule: FOL rule
        :param input_mapping: mapping between dataset features and logic variables
        :param output_mapping: mapping between classes and output logic constants
        :return: the approximated function
        """
        terms = self._parse(rule)
        tree = AST()
        for term in terms:
            tree.insert(term[0], term[1])
        return tree.root.call(input_mapping, output_mapping)

    def structure(self, rule: str, optimize: bool = False, implication: bool = False) -> Node:
        terms = self._parse(rule)
        tree = AST()
        for term in terms:
            tree.insert(term[0], term[1])
        if tree.root.logic_element == Skip:
            return tree.root
        elif implication:
            return tree.root.optimize() if optimize else tree.root
        else:
            return tree.root.children[0].optimize() if optimize else tree.root.children[0]

    @staticmethod
    def default_parser():
        return Parser([Variable, OutputVariable, OutputConstant, Equivalence, Conjunction, ReverseImplication, LeftPar,
                       RightPar, Implication, Disjunction, Plus, Negation, Constant, Product, NotEqual, Less,
                       DoubleImplication, LessEqual, Greater, GreaterEqual, Skip, OutputEquivalence])

    def _parse(self, string: str) -> list[tuple[LogicElement.__class__, Any]]:
        results = []
        string = string.replace(' ', '')
        while len(string) > 0:
            old_string = string
            for element in self.accepted_elements:
                match, value = element.parse(string)
                if match:
                    string = string.replace(value, '', 1)
                    results.append((element, value))
                    break
            if old_string == string:
                parse_exception(string)
        return results


class AST:

    def __init__(self, parent_ast: AST = None):
        self.root: Union[Node, None] = None
        self.tmp_ast: Union[AST, None] = None
        self.parent_ast: Union[AST, None] = parent_ast

    def __str__(self) -> str:
        return str(self.root)

    # TODO: think a better way to do this.
    def insert(self, lo: LogicElement.__class__, arg: Any) -> None:
        # If there is already an open left par call insert on the tmp ast
        if self.tmp_ast is not None:
            self.tmp_ast.insert(lo, arg)
        # If there is a right par close the tmp ast
        elif lo == RightPar and self.parent_ast is not None:
            # If the parent ast is not empty append this ast as the rightmost child of the first incomplete node
            if self.parent_ast.root is not None:
                incomplete_node = self.parent_ast.root._first_incomplete_node()
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
        elif self.root.logic_element == Variable or self.root.logic_element == Output:
            self.root = Node(lo, arg, [self.root])
        # If ast is not complete
        elif not self.root.is_complete:
            self.root.insert(lo, arg)
        # Add ol as new root if AST is complete and there is no priority
        elif self.root.is_complete and lo.priority <= self.root.logic_element.priority:
            self.root = Node(lo, arg, [self.root])
        # Change rightmost child if there is priority and AST is complete
        elif self.root.is_complete and lo.priority > self.root.logic_element.priority:
            rightmost_child = self.root.children[-1]
            new_node = Node(lo, arg, [rightmost_child], self.root)
            rightmost_child.father = new_node
            self.root.children[-1] = new_node


class Node:

    def __init__(self, logic_element: LogicElement.__class__, arg: Any, children: list = None, father: Node = None):
        self.logic_element: LogicElement.__class__ = logic_element
        self.arg = arg
        self.father = father
        self.children: list[Node] = [] if children is None else children

    def __str__(self, level: int = 0):
        result = '\t' * level + repr(self.arg) + '\n'
        return result + ''.join(child.__str__(level + 1) for child in self.children)

    # TODO: add a check on the index if possible
    def __getitem__(self, item, _count=0):
        """
        Nodes are indexed from root to leaves in leftmost depth search
        :param item: node's index
        :return: the corresponding node
        """
        if item == _count:
            return self
        else:
            for child in self.children:
                _count = _count + 1
                return child[item, _count]

    @property
    def __len__(self) -> int:
        return 1 + sum(len(child) for child in self.children) if len(self.children) > 0 else 1

    @property
    def depth(self) -> int:
        return 0 if self.father is None else 1 + self.father.depth

    def copy(self) -> Node:
        return Node(self.logic_element, self.arg, self.children, self.father)

    def optimize(self, _father=None) -> Node:
        if len(self.children) > 0:
            element = self.logic_element
            new_node = Node(element, self.arg, father=_father)
            new_children = [node.optimize(new_node) for node in self._fringe()]
            new_node.children = new_children
            return new_node
        else:
            new_node = self.copy()
            new_node.father = _father
            return new_node

    # TODO: refactor this mess
    def call(self, input_mapping: dict, output_mapping: dict = None) -> Callable:
        im, om = input_mapping, output_mapping
        if self.logic_element == Skip:
            return lambda x, y: self.logic_element(Variable(x), OutputVariable(y)).compute()
        elif self.logic_element == OutputVariable:
            return lambda x: self.logic_element(x).compute()
        elif self.logic_element == OutputConstant:
            return lambda _: self.logic_element(output_mapping[self.arg]).compute()
        elif self.logic_element == Constant:
            return lambda _: self.logic_element(self.arg).compute()
        elif self.logic_element == Variable:
            return lambda x: self.logic_element(x[:, input_mapping[self.arg]]).compute()
        elif self.logic_element.arity == 1:
            return lambda x: self.logic_element(self.children[0].call(im, om)(x)).compute()
        elif self.logic_element.arity == 2 and (self.logic_element == Implication or self.logic_element == DoubleImplication or
                                                self.logic_element == ReverseImplication):
            return lambda x, y: self.logic_element(self.children[0].call(im, om)(x),
                                                   self.children[1].call(im, om)(y)).compute()
        else:
            return lambda x: self.logic_element(self.children[0].call(im, om)(x), self.children[1].call(im, om)(x)).compute()

    @property
    def is_complete(self) -> bool:
        """
        A node is complete when the arity of its logic element is equal to its children and recursively all children are complete.
        """
        if self.logic_element == Variable or self.logic_element == Output:
            return True
        elif self.logic_element.arity == len(self.children):
            return all([node.is_complete for node in self.children])
        else:
            return False

    def insert(self, lo: LogicElement.__class__, arg: Any) -> None:
        done: bool = False
        for node in self.children:
            if done:
                break
            done = node._insert(lo, arg, done)
        if not done:
            self.children.append(Node(lo, arg, father=self))

    def _first_incomplete_node(self) -> Union[None, Node]:
        if self.logic_element.arity == 0:
            return None
        elif len(self.children) > 0:
            children_results = [node._first_incomplete_node() for node in self.children]
            for result in children_results:
                if result is not None:
                    return result
            return None if len(self.children) == self.logic_element.arity else self
        else:
            return self

    def _insert(self, lo: LogicElement.__class__, arg: Any, done: bool) -> bool:
        for node in self.children:
            if done:
                return True
            done = node._insert(lo, arg, done)

        if not done and len(self.children) < self.logic_element.arity:
            self.children.append(Node(lo, arg, father=self))
            return True
        elif not done:
            return False

    def _equal_operator_depth(self, result=0) -> int:
        if self.logic_element == self.children[0].logic_element:
            if len(self.children) == 2 and self.logic_element == self.children[1].logic_element:
                return result + self.children[0]._equal_operator_depth(result)\
                       + self.children[1]._equal_operator_depth(result) + 1
            else:
                return result + self.children[0]._equal_operator_depth(result) + 1
        else:
            return result + self.logic_element.arity

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

    def prune_leaves(self) -> None:
        if len(self.children) > 0:
            children_copy = self.children.copy()
            for node in children_copy:
                node.prune_leaves()
        else:
            self.father.children.remove(self)

    # TODO: is this really necessary?
    def prune_constants(self) -> None:
        if len(self.children) > 0:
            children_copy = self.children.copy()
            for node in children_copy:
                node.prune_constants()
        else:
            if self.logic_element == Constant:
                self.father.children.remove(self)

    def _fringe(self) -> list[Node]:
        if self.logic_element == self.children[0].logic_element:
            if len(self.children) == 2 and self.logic_element == self.children[1].logic_element:
                return self.children[0]._fringe() + self.children[1]._fringe()
            else:
                return self.children[0]._fringe() + [self.children[1]] if len(self.children) == 2 else self.children[0]._fringe()
        else:
            return self.children
