
import pytest
from simba import *
from simba.core import j_matrix
from sympy import Matrix, I, pprint, simplify, symbols, Rational

import simba.config
simba.config.params['checks'] = True


def test_j_matrix():
    assert j_matrix(4) == Matrix.diag(1, -1, 1, -1)


def test_extracting_coeffs_from_transfer_function():
    s = symbols('s')
    expr = (2 * s + 8) / (2 * s ** 2 + 4 * s + 2)
    coeffs = transfer_function_to_coeffs(expr, flip_s=False)
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


def test_unstable_filter_realisation():
    s = symbols('s')
    tf_expected = (s - 2) / (s + 2)
    ss = transfer_function_to_state_space(tf_expected)
    tf_result = ss.to_transfer_function()[0, 0]
    assert simplify(tf_result) == tf_expected, "Expected transfer function not recovered"
    ss = ss.extended_to_quantum()
    assert not ss.is_physically_realisable, "CCF state-space should not be physically realisable"

    # test that physically realisable unstable filter state space is realisable
    a = Matrix([[2, 0], [0, 2]])
    b = 2 * Matrix([[0, 1], [-1, 0]])
    c = 2 * Matrix([[0, -1], [1, 0]])
    d = Matrix.eye(2)
    assert StateSpace(a, b, c, d).is_physically_realisable,\
        "Unstable filter state-space should be realisable"

    # transform to physically realisable
    ss_2 = ss.to_physically_realisable()
    assert ss_2.is_physically_realisable, "Result should be physically realisable"

    s, k, r = ss_2.to_skr()
    assert s == Matrix.eye(2), "Expected identity scattering matrix"
    assert abs(k) == Matrix([[0, 2], [2, 0]]), "Did not get expected coupling matrix"
    assert r == Matrix.zeros(2), "Expected zero Hamiltonian matrix"


def test_unrealisable_transfer_function_should_raise_error():
    s = symbols('s')
    g = (s + I) / (s - I)
    ss = transfer_function_to_state_space(g).extended_to_quantum()
    with pytest.raises(StateSpaceError):
        ss.find_transformation_to_physically_realisable()

    assert not is_transfer_matrix_physically_realisable(g * Matrix.eye(2))


def test_finding_2_dof_realisation():
    s = symbols('s')
    # cascade of two tuned cavities
    tf = (s + 1)**2 / (s - 1)**2
    ss = transfer_function_to_state_space(tf).extended_to_quantum().to_physically_realisable()
    ss.pprint()
    gs, h_d = split_system(ss.to_slh())

    from simba.graph import nodes_from_dofs
    nodes = nodes_from_dofs(gs, h_d)

    assert len(nodes) == 2
    # should only have series collection
    for node in nodes:
        assert len(node.connections) == 0


@pytest.mark.slow
def test_finding_3_dof_realisation():
    s = symbols('s')
    tf = (s**3 + s**2 + s - 1) / (-s**3 + s**2 - s - 1)

    simba.config.params['checks'] = False
    ss = transfer_function_to_state_space(tf).extended_to_quantum().to_physically_realisable()
    nodes_from_dofs(*split_system(ss.to_slh()))
    simba.config.params['checks'] = True
    assert ss.is_physically_realisable


def test_recovering_transfer_function_for_cascade_realisation():
    s = symbols('s')
    # cascade of two tuned cavities
    tf = (s + 1)**2 / (s - 1)**2
    ss = transfer_function_to_state_space(tf)
    assert simplify(ss.to_transfer_function()[0, 0] - tf) == 0


def test_state_space_to_transfer_function_for_quantum_system():
    s = symbols('s')
    tf_expected = (s - 2) / (s + 2)
    ss = transfer_function_to_state_space(tf_expected).extended_to_quantum()
    assert simplify(ss.to_transfer_function() - Matrix.diag(tf_expected, tf_expected)) == Matrix.zeros(2), \
        "Expected transfer function not recovered"


def test_extending_to_quantum_state_space():
    ss = StateSpace(Matrix([1 + I]), Matrix([2]), Matrix([3]), Matrix([4]))
    ss = ss.extended_to_quantum()
    assert ss.a == Matrix.diag(1 + I, 1 - I)
    assert ss.c == Matrix.eye(2) * 3


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


def test_concatenation_product():
    s, a, b = symbols('s a b')
    tf = (s - 2) / (s + 2)
    system = transfer_function_to_state_space(tf).extended_to_quantum().to_physically_realisable()
    g_a, g_b = system.to_slh(a), system.to_slh(b)
    g_ab = concat(g_a, g_b)

    assert g_ab.s == Matrix.eye(4), "Expected identity scattering matrix"
    assert g_ab.k == Matrix.diag(g_a.k, g_a.k), "Expected block diagonal coupling matrix"
    assert g_ab.r == Matrix.zeros(4), "Expected zero Hamiltonian matrix"
