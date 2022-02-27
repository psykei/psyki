from typing import Callable
from tensorflow import SparseTensor, reduce_max, reshape
from tensorflow.python.keras.backend import to_dense, cast, maximum, minimum, constant, tile, shape
from psyki.utils import eta
from resources.dist.resources.DatalogParser import DatalogParser
from resources.dist.resources.DatalogVisitor import DatalogVisitor


class Fuzzifier(DatalogVisitor):

    def __init__(self, class_mapping: dict[str, int], feature_mapping: dict[str, int]):
        self.feature_mapping = feature_mapping
        self.class_mapping = {string: cast(to_dense(SparseTensor([[0, index]], [1.], (1, len(class_mapping)))), float)
                              for string, index in class_mapping.items()}
        self.classes: dict[str, Callable] = {}
        self._predicates: dict[str, Callable] = {}
        self.__rhs: dict[str, Callable] = {}

    def clause_expression(self, ctx: DatalogParser.ClauseExpressionContext or DatalogParser.ClauseExpressionNoParContext):
        l, r = self.visit(ctx.left), self.visit(ctx.right)
        operation = {
            '∧': lambda x: eta(maximum(l(x), r(x))),
            '∨': lambda x: eta(minimum(l(x), r(x))),
            '→': lambda x: eta(l(x) - r(x)),
            '↔': lambda x: eta(abs(l(x) - r(x))),
            '=': lambda x: eta(abs(l(x) - r(x))),
            '<': lambda x: eta(constant(1.) - eta(minimum(eta(constant(1.) - maximum(l(x) - r(x))),
                                                          eta(abs(l(x) - r(x)))))),
            '≤': lambda x: eta(constant(1.) - eta(constant(1.) - maximum(constant(0.), l(x) - r(x)))),
            '>': lambda x: eta(constant(1.) - maximum(constant(0.), l(x) - r(x))),
            '≥': lambda x: eta(minimum(eta(constant(1.) - maximum(l(x) - r(x))), eta(abs(l(x) - r(x))))),
            'm': lambda x: minimum(l(x), r(x)),
            '+': lambda x: l(x) + r(x),
            '*': lambda x: l(x) * r(x)
        }
        return operation.get(ctx.op.text)

    # Visit a parse tree produced by folParser#formula.
    def visitFormula(self, ctx: DatalogParser.FormulaContext):
        predicate, class_name = self.visit(ctx.lhs)
        r = self.visit(ctx.rhs)

        if class_name is not None:
            class_tensor = reshape(self.class_mapping[class_name], (1, len(self.class_mapping)))
            l = lambda y: eta(reduce_max(abs(tile(class_tensor, (shape(y)[0], 1)) - y), axis=1))
            if class_name not in self.classes.keys():
                # self.classes[class_name] = lambda x, y: eta(r(x) - l(y))
                self.classes[class_name] = lambda x, y: eta(l(y) - r(x))
                self.__rhs[class_name] = lambda x: r(x)
            else:
                incomplete_function = self.__rhs[class_name]
                self.classes[class_name] = lambda x, y: eta(l(y) - minimum(incomplete_function(x), r(x)))
                self.__rhs[class_name] = lambda x: minimum(incomplete_function(x), r(x))
        else:
            if predicate not in self._predicates.keys():
                self._predicates[predicate] = lambda x: r(x)
            else:
                incomplete_function = self._predicates[predicate]
                self._predicates[predicate] = lambda x: eta(minimum(incomplete_function(x), r(x)))

    # Visit a parse tree produced by DatalogParser#DefPredicateArgs.
    def visitDefPredicateArgs(self, ctx:DatalogParser.DefPredicateArgsContext):
        class_name = self.get_class_name(ctx.args)
        return ctx.pred.text, class_name

    # Visit a parse tree produced by PrologParser#LiteralPred.
    def visitLiteralPred(self, ctx: DatalogParser.LiteralPredContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by PrologParser#LiteralNeg.
    def visitLiteralNeg(self, ctx: DatalogParser.LiteralNegContext):
        return lambda x: eta(constant(1.) - self.visit(ctx.pred)(x))

    # Visit a parse tree produced by PrologParser#PredicateArgs.
    def visitPredicateArgs(self, ctx: DatalogParser.PredicateArgsContext):
        return self._predicates[ctx.pred.text]

    # Visit a parse tree produced by PrologParser#PredicateUnary.
    def visitPredicateUnary(self, ctx: DatalogParser.PredicateUnaryContext):
        return ctx.pred.text

    # Visit a parse tree produced by PrologParser#ClauseExpressionNoPar.
    def visitClauseExpressionNoPar(self, ctx: DatalogParser.ClauseExpressionNoParContext):
        return self.clause_expression(ctx)

    # Visit a parse tree produced by folParser#ClauseExpression.
    def visitClauseExpression(self, ctx: DatalogParser.ClauseExpressionContext):
        return self.clause_expression(ctx)

    # Visit a parse tree produced by DatalogParser#ConstName.
    def visitConstName(self, ctx: DatalogParser.ConstNameContext):
        return ctx.name.text

    # Visit a parse tree produced by PrologParser#ConstNumber.
    def visitConstNumber(self, ctx: DatalogParser.ConstNumberContext):
        return lambda _: float(ctx.num.text)

    # Visit a parse tree produced by folParser#TermVar.
    def visitTermVar(self, ctx: DatalogParser.TermVarContext):
        var = ctx.var.text
        return lambda x: x[:, self.feature_mapping[var]] \
            if var in self.feature_mapping.keys() else self.visitChildren(ctx)

    def get_class_name(self, args: DatalogParser.LastTermContext or DatalogParser.MoreArgsContext) -> str:
        if isinstance(args, DatalogParser.MoreArgsContext):
            return self.get_class_name(args.args)
        elif isinstance(args, DatalogParser.LastTermContext):
            result = self.visit(args.last)
            return result if isinstance(result, str) else None
