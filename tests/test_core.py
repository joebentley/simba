
import pytest
from simba import *
from sympy import Matrix, I


def test_1d_matrices_should_not_raise_error():
    one_d = Matrix([1])
    StateSpace(one_d, one_d, one_d, one_d)


def test_mismatched_matrices_should_raise_error():
    pair = Matrix([1, 2])
    with pytest.raises(DimensionError):
        StateSpace(pair, pair, pair, pair)


def test_three_dof_two_input_two_output_should_not_raise_error():
    a = Matrix.eye(3)
    b = Matrix([[1, 2], [3, 4], [5, 6]])
    c = Matrix([[1, 2, 3], [4, 5, 6]])
    d = Matrix.eye(2)
    StateSpace(a, b, c, d)


def test_should_error_if_transfer_function_coeffs_lists_are_wrong_length():
    # should not error
    StateSpace.from_transfer_function_coeffs([1, 2, 3], [2, 3])
    # should error
    with pytest.raises(CoefficientError):
        StateSpace.from_transfer_function_coeffs([1, 2], [3, 4])


def test_extending_to_quantum_state_space():
    ss = StateSpace(Matrix([1 + I]), Matrix([2]), Matrix([3]), Matrix([4]))
    ss = ss.extended_to_quantum()
    assert ss.a == Matrix.diag(1 + I, 1 - I)
    assert ss.c == Matrix.eye(2) * 3

    with pytest.raises(StateSpaceError):  # should raise error if already quantum
        ss.extended_to_quantum()


def test_truncating_to_classical_state_space():
    ss = StateSpace(Matrix([1 + I]), Matrix([2]), Matrix([3]), Matrix([4]))
    assert ss.extended_to_quantum().truncated_to_classical() == ss
