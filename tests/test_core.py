
import pytest
from simba import *
from sympy import Matrix, I, pprint, simplify, symbols


def test_extracting_coeffs_from_transfer_function():
    s = symbols('s')
    expr = (2 * s + 8) / (2 * s ** 2 + 4 * s + 2)
    coeffs = transfer_function_to_coeffs(expr)
    assert coeffs.numer == [4, 1, 0]
    assert coeffs.denom == [1, 2]

    with pytest.raises(NotImplementedError):
        transfer_function_to_coeffs(1 / expr)


def example_statespace():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([5, 6])
    c = Matrix([[7, 8]])
    d = Matrix([9])
    return StateSpace(a, b, c, d)


def example_two_input_two_output():
    a = Matrix.eye(3)
    b = Matrix([[1, 2], [3, 4], [5, 6]])
    c = Matrix([[1, 2, 3], [4, 5, 6]])
    d = Matrix.eye(2)
    return StateSpace(a, b, c, d)


def test_1d_matrices_should_not_raise_error():
    one_d = Matrix([1])
    StateSpace(one_d, one_d, one_d, one_d)


def test_mismatched_matrices_should_raise_error():
    pair = Matrix([1, 2])
    with pytest.raises(DimensionError):
        StateSpace(pair, pair, pair, pair)


def test_three_dof_two_input_two_output_should_not_raise_error():
    example_two_input_two_output()


def test_that_num_dof_inputs_output_given_correctly():
    ss = example_statespace()
    assert ss.num_degrees_of_freedom == 2
    assert ss.num_inputs == 1
    assert ss.num_outputs == 1


def test_should_error_if_transfer_function_coeffs_lists_are_wrong_length():
    # should not error
    StateSpace.from_transfer_function_coeffs([1, 2, 3], [2, 3])
    # should error
    with pytest.raises(CoefficientError):
        StateSpace.from_transfer_function_coeffs([1, 2], [3, 4])


def test_unstable_filter_to_state_space():
    s = symbols('s')
    tf_expected = (s - 2) / (s + 2)
    tf_result = transfer_function_to_state_space(tf_expected).to_transfer_function()
    assert simplify(tf_result) == tf_expected


def test_state_space_to_transfer_function_throws_expected_errors():
    ss = example_statespace()
    with pytest.raises(StateSpaceError):
        ss.extended_to_quantum().to_transfer_function()
    ss = example_two_input_two_output()
    with pytest.raises(DimensionError):
        ss.to_transfer_function()


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


def test_reordering_to_paired_form():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    c = Matrix([[9, 1], [2, 3]])
    d = Matrix([[4, 5], [6, 7]])
    a, b, c, d = StateSpace(a, b, c, d).extended_to_quantum().reorder_to_paired_form()

    assert a == Matrix([[1, 0, 2, 0], [0, 1, 0, 2], [3, 0, 4, 0], [0, 3, 0, 4]])
    assert b == Matrix([[5, 0, 6, 0], [0, 5, 0, 6], [7, 0, 8, 0], [0, 7, 0, 8]])
    assert c == Matrix([[9, 0, 1, 0], [0, 9, 0, 1], [2, 0, 3, 0], [0, 2, 0, 3]])
    assert d == Matrix([[4, 0, 5, 0], [0, 4, 0, 5], [6, 0, 7, 0], [0, 6, 0, 7]])
