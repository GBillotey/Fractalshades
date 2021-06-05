# -*- coding: utf-8 -*-
import numpy as np
import numbers
import re

import ast
import copy

safe_names = ["x", "np"]
safe_attrs = ["sin", "cos", "tan", "arccos", "arcsin", "arctan", "exp", "log",
              "sqrt", "isin", "where"]

def acceptable_expr(expr):

    if isinstance(expr, ast.Expression):
        return acceptable_expr(expr.body)

    elif isinstance(expr, ast.Num):
        return True

    elif isinstance(expr, ast.BinOp):
        return (acceptable_expr(expr.left)
                and acceptable_expr(expr.right))
    
    elif isinstance(expr, ast.Compare):
        return (acceptable_expr(expr.left)
                and all(acceptable_expr(cmp) for cmp in expr.comparators))

    elif isinstance(expr, ast.Lambda):
        return (acceptable_expr(expr.args)
                and acceptable_expr(expr.body))

    elif isinstance(expr, ast.arguments):
        print("arguments", expr.__dict__)
        return all(acceptable_expr(arg) for arg in expr.args)

    elif isinstance(expr, ast.arg):
        print("arg", expr.__dict__)
        return expr.arg == "x"

    elif isinstance(expr, ast.Name):
        print("name", expr.__dict__)
        return expr.id in safe_names

    elif isinstance(expr, ast.Attribute):
        print("attribute", expr.__dict__)
        return (acceptable_expr(expr.value)
                and (expr.attr in safe_attrs))

    elif isinstance(expr, ast.Call): 
        print("call", expr.__dict__)
        return (acceptable_expr(expr.func)
                and all(acceptable_expr(arg) for arg in expr.args))

    elif (isinstance(expr, ast.List) or isinstance(expr, ast.Tuple)):
        return all(acceptable_expr(arg) for arg in expr.elts)

    print(expr)
    return False

def safe_eval(expr):
    e = ast.parse(expr, mode="eval")
    if acceptable_expr(e):
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

    f = safe_eval("lambda x: np.sin(x) + np.isin(x, (1, 2, 3)) * np.where(x==2, 0., x**x)")
    print(f)
    # print(f(1))

#    e = ast.parse("lambda x: 2. * x", mode="eval")
#    f = safe_eval(e)
#    print(f)
#    print(f(1))
