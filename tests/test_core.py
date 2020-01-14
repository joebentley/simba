
import pytest
from simba import *
import sympy


def test_1d_matrices_should_not_raise_error():
    one_d = sympy.Matrix([1])
    StateSpace(one_d, one_d, one_d, one_d)


def test_mismatched_matrices_should_raise_error():
    pair = sympy.Matrix([1, 2])
    with pytest.raises(DimensionError):
        StateSpace(pair, pair, pair, pair)


def test_three_dof_two_input_two_output_should_not_raise_error():
    a = sympy.eye(3)
    b = sympy.Matrix([[1, 2], [3, 4], [5, 6]])
    c = sympy.Matrix([[1, 2, 3], [4, 5, 6]])
    d = sympy.eye(2)
    StateSpace(a, b, c, d)
