from typing import Union, Any, Callable
from psyki.logic import LogicOperator, L, LT, LeftPar, RightPar, Implication, LTX, LTY


class AST:

    def __init__(self, parent_ast=None):
        self.root: Union[Node, None] = None
        self.tmp_ast: Union[AST, None] = None
        self.parent_ast: Union[AST, None] = parent_ast

    def insert(self, lo: LogicOperator.__class__, arg: Any):
        # If there is already an open left par call insert on the tmp ast
        if self.tmp_ast is not None:
            self.tmp_ast.insert(lo, arg)
        # If there is a right par close the tmp ast
        elif lo == RightPar and self.parent_ast is not None:
            # If the parent ast is not empty append this ast as the rightmost child
            if self.parent_ast.root is not None:
                self.parent_ast.root.children.append(self.root)
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
            self.root.children[-1] = Node(lo, arg, [rightmost_child])


class Node:

    def __init__(self, lo: LogicOperator.__class__, arg: Any, children: list = None):
        self.operator: LogicOperator.__class__ = lo
        self.arg = arg
        self.children: list[Node] = [] if children is None else children

    def call(self, input_mapping: dict, output_mapping: dict) -> Callable:
        im, om = input_mapping, output_mapping
        if self.operator == L:
            return lambda x: self.operator(x[input_mapping[self.arg]]).accept()
        elif self.operator == LTX:
            return lambda x: self.operator(x).accept()
        elif self.operator == LTY:
            return lambda _: self.operator(output_mapping[self.arg]).accept()
        elif self.operator.arity == 1:
            return lambda x: L(self.operator(self.children[0].call(im, om)(x)).accept())
        elif self.operator.arity == 2 and (self.operator == Implication):
            return lambda x, y: self.operator(self.children[0].call(im, om)(x), self.children[1].call(im, om)(y)).accept()
        else:
            return lambda x: L(self.operator(self.children[0].call(im, om)(x), self.children[1].call(im, om)(x)).accept())

    def is_complete(self) -> bool:
        if self.operator == L or self.operator == LT:
            return True
        elif self.operator.arity == len(self.children):
            return all([node.is_complete() for node in self.children])
        else:
            return False

    def insert(self, lo: LogicOperator.__class__, arg: Any):
        done: bool = False
        for node in self.children:
            if done:
                break
            done = node._insert(lo, arg, done)
        if not done:
            self.children.append(Node(lo, arg))

    def _insert(self, lo: LogicOperator.__class__, arg: Any, done: bool) -> bool:
        for node in self.children:
            if done:
                return True
            done = node._insert(lo, arg, done)

        if not done and len(self.children) < self.operator.arity:
            self.children.append(Node(lo, arg))
            return True
        elif not done:
            return False
