from simba.errors import DimensionError
from sympy import Matrix


def halve_matrix(mat):
    """
    Halve the dimensions of the given matrix, truncating where necessary.

    Throws `DimensionError` if matrix dimensions are not even.
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
    return list(map(lambda sol: Matrix(sol).reshape(*x.shape), sols))


def construct_transformation_matrix(n):
    """Construct permutation matrix that reorders the elements so that (1, 2, 3, 11, 22, 33) -> (1, 11, 2, 22, 3, 33)"""
    if n % 2 != 0:
        raise DimensionError("n should be even")

    u = Matrix.zeros(n, n)
    for x in range(n // 2):
        u[x * 2, x] = 1
        u[x * 2 + 1, x + n // 2] = 1
    return u


def matrix_simplify(m):
    """Try to quickly simplify matrix m."""
    from sympy import radsimp, powsimp, expand, Matrix, simplify

    m = Matrix(m)  # make a mutable copy

    rows, cols = m.shape
    for i in range(rows):
        for j in range(cols):
            m[i, j] = simplify(expand(radsimp(powsimp(expand(m[i, j])), symbolic=False)))

    return m
