from typing import Union
from psyki.logic import LogicOperation, L, LT, LeftPar, RightPar


class AST:

    def __init__(self, parent_ast=None):
        self.root: Union[Node, None] = None
        self.tmp_ast: Union[AST, None] = None
        self.parent_ast: Union[AST, None] = parent_ast

    def insert(self, lo: LogicOperation):
        # If there is an open left par call insert on the tmp ast
        if self.tmp_ast is not None:
            self.tmp_ast.insert(lo)
        # If there is a right par close the tmp ast
        elif isinstance(lo, RightPar) and self.parent_ast is not None:
            self.parent_ast.root.children.append(self.root)
            self.parent_ast.tmp_ast = None
        # Create a new tmp ast if there is a left par
        elif isinstance(lo, LeftPar):
            self.tmp_ast = AST(self)
        # Base case: empty AST
        elif self.root is None:
            self.root = Node(lo)
        # If ast is not complete
        elif not self.root.is_complete():
            self.root.insert(lo)
        # If there is a variable in the root change it with the new lo, the variable become a child
        elif isinstance(self.root.operator, L) or isinstance(self.root.operator, LT):
            old_ast = self.root.children.copy().append(self.root)
            self.root.children = []
            self.root = Node(lo, old_ast)
        # Add ol as new root if AST is complete and there is no priority
        elif self.root.is_complete() and lo.priority <= self.root.operator.priority:
            self.root = Node(lo, [self.root])
        # Change rightmost child if there is priority and AST is complete
        elif self.root.is_complete() and lo.priority > self.root.operator.priority:
            rightmost_child = self.root.children[-1]
            self.root.children[-1] = Node(lo, [rightmost_child])


class Node:

    def __init__(self, lo: LogicOperation, children: list = None):
        self.operator: LogicOperation = lo
        self.children: list[Node] = [] if children is None else children

    def is_complete(self) -> bool:
        if isinstance(self.operator, L) or isinstance(self.operator, LT):
            return True
        elif self.operator.arity == len(self.children):
            return all([node.is_complete() for node in self.children])
        else:
            return False

    def insert(self, lo: LogicOperation):
        done: bool = False
        for node in self.children:
            if done:
                break
            done = node._insert(lo, done)
        if not done:
            self.children.append(Node(lo))
        # raise Exception(self.operator.name + ' accept exactly ' + str(self.operator.arity) + 'arguments.')

    def _insert(self, lo: LogicOperation, done: bool) -> bool:
        for node in self.children:
            if done:
                return True
            done = node._insert(lo, done)

        if not done and len(self.children) < self.operator.arity:
            self.children.append(Node(lo))
            return True
        elif not done:
            return False
