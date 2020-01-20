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
