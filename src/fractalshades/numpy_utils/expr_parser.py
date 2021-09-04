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
    e = ast.parse(expr, mode="eval")
    if acceptable_expr(e, safe_vars=variables): #, safe_attrs=safe_attrs):
        return eval(expr)
    else:
        return None


if __name__ == "__main__":
#    print(safe_eval("3 + 5 * 2"))
#
#    f = safe_eval("lambda x: x * 2.")
#    print(f)
#    print(f(1))
#
#    f = safe_eval("lambda x: np.sin(x)")
#    print(f)

    f = func_parser(["x"], "np.sin(x) + np.isin(x, (1, 2, 3)) * np.where(x==2, 0., x**x)")
    print(f)
    print(f(0))
    f = func_parser(["x1", "x2", "x3"], "x1 + x2 + x3")
    print(f)
    print(f(1, 2, 3))
    # print(f(1))

#    e = ast.parse("lambda x: 2. * x", mode="eval")
#    f = safe_eval(e)
#    print(f)
#    print(f(1))
