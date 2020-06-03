import simba.config as conf
from simba.errors import DimensionError
import sympy


def halve_matrix(mat: sympy.Matrix):
    """
    Halve the dimensions of the given matrix, truncating where necessary.

    Throws `DimensionError` if matrix dimensions are not even.

    Example:
        >>>from sympy import Matrix
        >>>m = Matrix([[1, 2], [3, 4]])
        >>>assert halve_matrix(m) == Matrix([[1]])
    """
    if mat.shape[0] % 2 != 0 or mat.shape[1] % 2 != 0:
        raise DimensionError(f"Matrix dimensions not even: {mat.shape}")

    return mat.extract(range(mat.shape[0] // 2), range(mat.shape[1] // 2))


def solve_matrix_eqn(eqn, x):
    """
    Solve matrix eqn for x, where eqn is a matrix equation or list of matrix equations (assumed equal to zero on RHS)
    and x is a sympy ``MatrixSymbol`` object.

    Transforms all solutions to list of matrices (same shape as x).
    """
    from sympy import linsolve

    if isinstance(eqn, list):
        eqns = []
        for e in eqn:
            eqns.extend(list(e))  # make single list of all equations
    else:
        eqns = list(eqn)

    sols = linsolve(eqns, list(x))
    return list(map(lambda sol: sympy.Matrix(sol).reshape(*x.shape), sols))


def construct_permutation_matrix(n: int) -> sympy.Matrix:
    """Construct permutation matrix that reorders the elements so that (1, 2, 3, 11, 22, 33) -> (1, 11, 2, 22, 3, 33)"""
    if n % 2 != 0:
        raise DimensionError("n should be even")

    u = sympy.Matrix.zeros(n, n)
    for x in range(n // 2):
        u[x * 2, x] = 1
        u[x * 2 + 1, x + n // 2] = 1
    return u


def simplify(expr, rhs=None):
    """Simplify given expression or equation, using wolframscript if config.params['wolframscript'] is True"""
    if conf.params['wolframscript']:
        import subprocess
        from sympy import Eq
        import sympy.printing.mathematica as m
        import sympy.parsing.mathematica as mp

        if rhs is not None:
            s = m.mathematica_code(Eq(expr, rhs, evaluate=False))
        else:
            s = m.mathematica_code(expr)

        result = subprocess.run(["wolframscript", "-code", f"Simplify[{s}]"], capture_output=True)

        return mp.mathematica(str(result.stdout))
    else:
        from sympy import simplify

        if rhs is not None:
            return simplify(expr) == rhs
        else:
            return simplify(expr)


def adiabatically_eliminate(expr: sympy.Expr, gamma: sympy.Symbol) -> sympy.Expr:
    """
    Eliminate terms from ``expr`` which are very small compared to ``gamma`` which is much larger than
    any other frequency.
    """
    if expr == 0:
        return sympy.Number(0)

    from sympy import fraction, Symbol

    numer, denom = fraction(expr)
    epsilon = Symbol('epsilon')
    numer = (numer / gamma).subs(gamma, 1 / epsilon).expand().subs(epsilon, 0)
    denom = (denom / gamma).subs(gamma, 1 / epsilon).expand().subs(epsilon, 0)
    return numer / denom
