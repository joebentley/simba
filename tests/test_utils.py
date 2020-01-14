import pytest
from simba.errors import DimensionError
from simba.utils import *
from sympy.matrices import Matrix


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
