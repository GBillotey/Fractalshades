# -*- coding: utf-8 -*-
import ast
import numpy as np  # This import IS needed

SAFE_NAMES = ["np"]
SAFE_ATTRS = ["sin", "cos", "tan", "arccos", "arcsin", "arctan", "exp", "log",
              "sqrt", "isin", "where", "pi", "mod"]


def acceptable_expr(expr, safe_vars):
    safe_names = safe_vars + SAFE_NAMES
    safe_attrs = SAFE_ATTRS

    def inner(expr):
        """
        Return True if "lambda x: " + `expr` is a valid function of one var, x.
        """
        if isinstance(expr, ast.Expression):
            return inner(expr.body)

        elif isinstance(expr, ast.Num):
            return True
        
        elif isinstance(expr, ast.UnaryOp):
            return inner(expr.operand)

        elif isinstance(expr, ast.BinOp):
            return (inner(expr.left) and inner(expr.right))

        elif isinstance(expr, ast.Compare):
            return (inner(expr.left)
                    and all(inner(cmp) for cmp in expr.comparators))

        elif isinstance(expr, ast.Lambda):
            return (inner(expr.args) and inner(expr.body))

        elif isinstance(expr, ast.arguments):
            return all(inner(arg) for arg in expr.args)

        elif isinstance(expr, ast.arg):
            return expr.arg in safe_vars

        elif isinstance(expr, ast.Name):
            return expr.id in safe_names

        elif isinstance(expr, ast.Attribute):
            return (inner(expr.value)
                    and (expr.attr in safe_attrs))

        elif isinstance(expr, ast.Call):
            return (inner(expr.func)
                    and all(inner(arg) for arg in expr.args))

        elif (isinstance(expr, ast.List) or isinstance(expr, ast.Tuple)):
            return all(inner(arg) for arg in expr.elts)

        return False

    return inner(expr)


def func_parser(variables, expr):
    """
    variables : array of strings, eg: ["x"], ["x1", "x2", "x3"], ["x", "y"]
                For safety, string should have max 2 chars
    expr : str, numpy function of the `variables` e.g. "np.sin(x1) + x2"
    Returns
    the associated function, if safe
    """
    for var in variables:
        if len(var) > 2:
            raise ValueError("Variable {} is more than 2 chars".format(var))
    expr = "lambda " + ", ".join(variables) + ": " + expr

    try:
        e = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    if acceptable_expr(e, safe_vars=variables): #, safe_attrs=safe_attrs):
        try:
            return eval(expr)
        except SyntaxError:
            return None
    else:
        return None

class Numpy_expr:
    # Note: We *could* use typing.Annotated when dropping 3.8 support
    # See https://docs.python.org/3/library/typing.html#typing.Annotated
    # https://peps.python.org/pep-0593/
    # T1 = Annotated[Numpy_expr, Vars("x", "y")]
    def __init__(self, variables, expr):
        f"""
        Defines a numpy expression from a string ; can be used to parse a
        GUI user-input string.

        Parameters:
        -----------
        variables: str or str[]
            The list of variables used in expr. The length of variables names
            shall be max 2 chars, e.g, "x", "x1", "x2", "y" "z", "xy".
        expr: str
            The numerical expression to be evaluated. The standard operations 
            (+, -, *, /, **) are accepted, and a subset of numpy
            functions: {SAFE_ATTRS}.
        """
        if isinstance(variables, str):
            variables = [variables]
        self.variables = variables
        self.expr = expr
        self._func = func_parser(variables, expr) 

    def validates(self):
        """
        Returns True if the Numpy_expr is valid
        """
        return self._func is not None

    @staticmethod
    def validates_expr(variables, expr):
        """ Static version - used in GUI validation """
        return func_parser(variables=variables, expr=expr) is not None
    
    def __str__(self):
        return self.expr

    def __call__(self, *args):
        if self._func is None:
            return args
        return self._func(*args)

    @property
    def init_kwargs(self):
        """ Return a dict of parameters used during __init__ call"""
        init_kwargs = {
            "variables": self.variables,
            "expr": self.expr,
        }
        return init_kwargs

    def __reduce__(self):
        """ Serialisation of a Numpy_expr object.
        """
        vals = tuple(self.init_kwargs.values())
        return (self.__class__, vals)


class Vars:
    # Could be used with Numpy_expr, see note above
    # (typing.Annotated new in Python 3.9)
    def __init__(self, *args):
        self.args = args

