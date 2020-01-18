import pytest
from simba.errors import DimensionError
from simba.utils import halve_matrix, solve_matrix_eqn
from simba.core import j_matrix
from sympy import Rational
from sympy.matrices import Matrix, MatrixSymbol


def test_halving_matrix():
    m = Matrix([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [8, 7, 5, 6],
                [4, 3, 2, 1]])
    assert halve_matrix(m) == Matrix([[1, 2], [5, 6]])

    m = Matrix([[1, 2],
                [3, 4],
                [5, 6],
                [7, 8]])

    assert halve_matrix(m) == Matrix([[1], [3]])

    with pytest.raises(DimensionError):  # should raise error if odd dimensions
        halve_matrix(Matrix.eye(3))


def test_solving_matrix_eqn():
    # unstable filter
    eye = Matrix.eye(2)
    a = 2 * eye
    b = eye
    c = 4 * eye
    d = b
    j = j_matrix(2)
    x = MatrixSymbol('X', *j.shape)

    sol = solve_matrix_eqn(a * x + x * a.H + b * j * b.H, x)
    assert len(sol) == 1 and sol[0] == Rational(-1, 4) * j
