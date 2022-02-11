from __future__ import annotations
import itertools
from typing import Iterable, Any, Union, Callable
from psyki.logic.elements import *
from psyki.utils import parse_exception


class Parser:
    """
    A parser processes a string logic rule.
    It can be used for computing a callable approximated function
    or to generate a data structure resembling that function (i.e. a tree)
    """

    def __init__(self, accepted_elements: Iterable[LogicElement.__class__]):
        self.accepted_elements = accepted_elements

    def function(self, rule: str, input_mapping: dict, output_mapping: dict) -> Callable:
        """
        Return a fuzzy function that can be invoked by passing the required input
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

    def structure(self, rule: str, optimize: bool = False) -> Node:
        """
        Return a (possibly optimized) tree representing the fuzzy function of the FOL rule
        :param rule: FOL rule
        :param optimize: true enable optimization
        :return: the approximated function in form of a tree
        """
        terms = self._parse(rule)
        tree = AST()
        for term in terms:
            tree.insert(term[0], term[1])
        if tree.root.logic_element == Skip:
            return tree.root
        else:
            return tree.root.children[0].optimize() if optimize else tree.root.children[0]

    @staticmethod
    def default_parser():
        """
        Create a parser capable of process all logic function and elementary algebraic operators.
        :return: a parser
        """
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
        self._tmp_ast: Union[AST, None] = None
        self._parent_ast: Union[AST, None] = parent_ast

    def __str__(self) -> str:
        return str(self.root)

    # TODO: think a better way to do this.
    def insert(self, lo: LogicElement.__class__, arg: Any) -> None:
        # If there is already an open left par call insert on the tmp ast
        if self._tmp_ast is not None:
            self._tmp_ast.insert(lo, arg)
        # If there is a right par close the tmp ast
        elif lo == RightPar and self._parent_ast is not None:
            # If the parent ast is not empty append this ast as the rightmost child of the first incomplete node
            if self._parent_ast.root is not None:
                incomplete_node = self._parent_ast.root._first_incomplete_node()
                if incomplete_node is None:
                    self._parent_ast.root.children.append(self.root)
                    self.root.father = self._parent_ast.root
                else:
                    incomplete_node.children.append(self.root)
                    self.root.father = incomplete_node
            else:
                self._parent_ast.root = self.root
            self._parent_ast._tmp_ast = None
        # Create a new tmp ast if there is a left par
        elif lo == LeftPar:
            self._tmp_ast = AST(self)
        # Base case: empty AST
        elif self.root is None:
            self.root = Node(lo, arg)
        # If there is a variable in the root change it with the new lo, the variable become a child
        elif self.root.logic_element == Variable or issubclass(self.root.logic_element, Variable):
            self.root = Node(lo, arg, [self.root])
        # If ast is not complete
        elif not self.root.is_complete:
            self.root.insert(lo, arg)
        # Add element as new root if AST is complete and there is no priority
        elif self.root.is_complete and lo.priority <= self.root.logic_element.priority:
            self.root = Node(lo, arg, [self.root])
        # Change rightmost child if there is priority and AST is complete
        elif self.root.is_complete and lo.priority > self.root.logic_element.priority:
            rightmost_child = self.root.children[-1]
            new_node = Node(lo, arg, [rightmost_child], self.root)
            rightmost_child.father = new_node
            self.root.children[-1] = new_node


class Node:

    def __init__(self, logic_element: LogicElement.__class__, arg: str, children: list = None, father: Node = None):
        self.logic_element: LogicElement.__class__ = logic_element
        self.arg = arg
        self.father = father
        self.children: list[Node] = [] if children is None else children

    def __str__(self, level: int = 0):
        result = '\t' * level + repr(self.arg) + '\n'
        return result + ''.join(child.__str__(level + 1) for child in self.children)

    def __hash__(self) -> int:
        return hash(str(self.logic_element)) + hash(self.arg) + hash(self.father)\
               + sum(hash(child) for child in self.children)

    def __eq__(self, other: Node) -> bool:
        return self.logic_element == other.logic_element and self.father == other.father\
               and self.children == other.children

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
            new_children = [node.optimize(new_node) for node in self._fringe]
            new_node.children = new_children
            return new_node
        else:
            new_node = self.copy()
            new_node.father = _father
            return new_node

    # TODO: think of a refactor for this function
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
        elif self.logic_element.arity == 2 and (self.logic_element == Implication or
                                                self.logic_element == DoubleImplication or
                                                self.logic_element == ReverseImplication):
            return lambda x, y: self.logic_element(self.children[0].call(im, om)(x),
                                                   self.children[1].call(im, om)(y)).compute()
        else:
            return lambda x: self.logic_element(self.children[0].call(im, om)(x),
                                                self.children[1].call(im, om)(x)).compute()

    @property
    def is_complete(self) -> bool:
        """
        A node is complete when the arity of its logic element is equal to its children
        and recursively all children are complete.
        """
        if self.logic_element == Variable or self.logic_element == Output:
            return True
        elif self.logic_element.arity == len(self.children):
            return all([node.is_complete for node in self.children])
        else:
            return False

    def insert(self, element: LogicElement.__class__, arg: str) -> None:
        """
        Insert a logic element into the tree
        :param element: the logic element
        :param arg: additional argument that characterize the element (e.g. variable name, constant value)
        """
        done: bool = False
        for node in self.children:
            if done:
                break
            done = node._insert(element, arg, done)
        if not done:
            self.children.append(Node(element, arg, father=self))

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

    @property
    def leaves(self) -> Iterable[Node]:
        """
        :return: all the leaves of the tree
        """
        if len(self.children) == 0:
            return self,
        else:
            return list(itertools.chain.from_iterable(child.leaves for child in self.children))

    def prune_leaves(self) -> None:
        """
        Prune all leaves of the tree
        """
        if len(self.children) > 0:
            children_copy = self.children.copy()
            for node in children_copy:
                node.prune_leaves()
        else:
            self.father.children.remove(self)

    @property
    def _fringe(self) -> list[Node]:
        """
        :return: the list of nodes that can be merged in a single one node
        """
        # If the leftmost child's element is the same of the current node
        if self.logic_element == self.children[0].logic_element:
            # If its a binary element and also the right child holds the same element return them
            if len(self.children) == 2 and self.logic_element == self.children[1].logic_element:
                return self.children[0]._fringe + self.children[1]._fringe
            else:
                return self.children[0]._fringe + [self.children[1]] \
                    if len(self.children) == 2 else self.children[0]._fringe
        else:
            return self.children
