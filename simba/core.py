
from copy import deepcopy
from typing import List, NamedTuple, Optional, Union, Tuple

import sympy

from simba.utils import solve_matrix_eqn, construct_permutation_matrix, simplify
from simba.errors import DimensionError, CoefficientError, StateSpaceError, ResultError
import simba.config as config

"""State-spaces and transfer functions"""


def is_transfer_matrix_physically_realisable(expr: sympy.Matrix, d_matrix: sympy.Matrix = None) -> bool:
    r"""
    Check if transfer matrix :math:`\mathbf{G}(s)` given by ``expr`` and direct-feed matrix ``d_matrix``
    is possible to physically realise by the conditions given in [transfer-function]_, i.e. that it obeys,

    .. math::
        \mathbf{G}^\sim(s) J \mathbf{G}(s) = J,\ \ D J D^\dagger = J,

    where :math:`\mathbf{G}^\sim(s) \equiv \mathbf{G}^\dag(-s^*)`.

    :param expr: a sympy Matrix representing the transfer Matrix :math:`\mathbf{G}(s)`
    :param d_matrix: a sympy Matrix representing the direct-feed Matrix, defaulting to the identity matrix
    """
    from sympy import conjugate, Symbol, Matrix

    j = j_matrix(expr.shape[0])
    if not d_matrix:
        d_matrix = Matrix.eye(*j.shape)

    s = Symbol('s')
    cond1 = simplify(expr.subs(s, -conjugate(s)).H * j * expr - j, Matrix.zeros(*j.shape))
    cond2 = simplify(d_matrix * j * d_matrix.H - j, Matrix.zeros(*j.shape))

    return cond1 and cond2


def transfer_func_coeffs_to_state_space(numer: List, denom: List) -> 'StateSpace':
    """See `StateSpace.from_transfer_function_coeffs`."""
    return StateSpace.from_transfer_function_coeffs(numer, denom)


def transfer_function_to_state_space(expr: sympy.Expr) -> 'StateSpace':
    """See `StateSpace.from_transfer_function`."""
    return StateSpace.from_transfer_function(expr)


def tf2ss(expr: sympy.Expr) -> 'StateSpace':
    """See `transfer_function_to_state_space`"""
    return transfer_function_to_state_space(expr)


def transfer_function_to_realisable_state_space(expr: sympy.Expr) -> 'StateSpace':
    """Convert given transfer function to physically realisable state space if possible."""
    return transfer_function_to_state_space(expr).extended_to_quantum().to_physically_realisable()


def tf2rss(expr: sympy.Expr) -> 'StateSpace':
    """See `transfer_function_to_realisable_state_space`"""
    return transfer_function_to_realisable_state_space(expr)


def tf2network(expr: sympy.Expr) -> 'SplitNetwork':
    """Calculate split network directly from transfer function"""
    rs = tf2rss(expr)
    slh = rs.to_slh()
    return slh.split()


class StateSpace:
    r"""
    Represents a dynamical quantum state-space which describes the time-domain evolution of a system.

    .. math::
        \dot{x} &= a x + b u, \\
        y &= c x + d u.

    where the state vectors are bounded linear operators usually in `doubled-up form`, (use `reorder_to_paired_form`
    to convert to `paired operator form`)

    .. math::
        x \in \mathbb{L}^{n\times 1},\
        u \in \mathbb{L}^{m\times 1},\
        y \in \mathbb{L}^{l\times 1},

    and, the "system matrices" are

    .. math::
        a \in \mathbb{C}^{n\times n},\
        b \in \mathbb{C}^{n\times m},\
        c \in \mathbb{C}^{l\times n},\
        d \in \mathbb{C}^{l\times m}.

    If the dimensions do not match up, raises `DimensionError`.

    For `SISO` quantum systems :math:`m = l = 2` (usually the operator and its conjugate operator,
    e.g. :math:`u = (\hat{u}, \hat{u}^\dagger)^T`).

    Attributes:
         - a:     Internal dynamics
         - b:     Input coupling
         - c:     Internal coupling to output
         - d:     Direct-feed (input to output)
         - paired_operator_form:  Whether or not system is in `paired operator form`.

    Note that the matrices are stored using ``sympy.ImmutableMatrix``, and so are immutable. New StateSpaces should
    be created from modified matrices instead.
    """
    def __init__(self, a, b, c, d, *, paired_operator_form=False):
        from sympy import ImmutableMatrix
        self.a = ImmutableMatrix(a)
        self.b = ImmutableMatrix(b)
        self.c = ImmutableMatrix(c)
        self.d = ImmutableMatrix(d)

        # check matrix dimensions
        if not a.is_square:
            raise DimensionError(f"`a` matrix is not square: {a.shape}")
        na = a.shape[0]
        (nb, mb) = b.shape
        (lc, nc) = c.shape
        (ld, md) = d.shape
        if not mb == md:
            raise DimensionError(f"Number of input channels for matrix b not equal to matrix d: {mb} != {md}")
        if not nb == na or not nc == na:
            raise DimensionError(f"Number of degrees of freedom in matrices do not match: "
                                 f"a: {na}, b: {nb}, c: {nc}")
        if not lc == ld:
            raise DimensionError(f"Number of output channels for matrix c not equal to matrix d: {lc} != {ld}")

        self.paired_operator_form = paired_operator_form  # set if the StateSpace is in `paired operator form`

    @classmethod
    def from_transfer_function_coeffs(cls, numer: List, denom: List) -> 'StateSpace':
        r"""
        Return the `SISO` controllable canonical form state space for the given list of numerators
        and denominators of a pole-zero form transfer function, given in order of ascending powers, assuming complex
        ladder operators :math:`(a, a^\dagger)` are used.

        The coefficients are defined via the transfer function between the input :math:`u(s)` and
        the output :math:`y(s)`, where :math:`s` is the complex Laplace frequency, [#laplace]_

        .. math::
            \frac{y(s)}{u(s)} = \frac{b_0 s^n + b_1 s^{n-1} + \dots + b_{n-1} s + b_n}
            {s^n + a_1 s^{n-1} + \dots + a_{n-1} s + a_n}.

        *Note that we assume the coefficients are normalized with respect to the highest order term in the denominator.*

        Raises `CoefficientError` if lengths of coefficient lists are wrong.

        Reference: https://www.engr.mun.ca/~millan/Eng6825/canonicals.pdf

        TODO: implement for MIMO systems

        :param numer: The numerator coefficients: :math:`[b_n, \dots, b_0]`
        :param denom: The denominator coefficients: :math:`[a_n, \dots, a_1]`

        :return: StateSpace for the given system
        """
        from sympy import Matrix

        if not len(denom) == len(numer) - 1:
            raise CoefficientError(f"Denominator coefficients list did not have length of numerator list minus one: "
                                   f"len(denom) == {len(denom)}, len(numer) == {len(numer)}")

        # construct b matrix
        n = len(denom)  # num. degrees of freedom
        b = Matrix.zeros(n, 1)  # single-input, 1 in last row
        b[n - 1] = 1

        # construct a matrix
        a = Matrix.zeros(n)
        for i in range(0, n - 1):
            a[i, i + 1] = 1
        for i, denom_coeff in enumerate(denom):
            a[n - 1, i] = -denom_coeff

        # construct c matrix
        b_0 = numer[-1]

        c = Matrix.zeros(1, n)
        for i, (a_i, b_i) in enumerate(zip(denom, numer)):
            c[0, i] = b_i - a_i * b_0

        # construct d matrix
        d = Matrix([b_0])

        return cls(a, b, c, d)

    @classmethod
    def from_transfer_function(cls, expr: sympy.Expr) -> 'StateSpace':
        """Call `from_transfer_function_coeffs` passing the expression to `transfer_function_to_coeffs`."""
        return cls.from_transfer_function_coeffs(*transfer_function_to_coeffs(expr))

    def to_transfer_function(self) -> sympy.Expr:
        """Calculate transfer function matrix for the system using the convention given by [#laplace]_."""
        from sympy import Symbol, Matrix
        s = Symbol('s')
        return self.c * (-s * Matrix.eye(self.a.shape[0]) - self.a).inv() * self.b + self.d

    def extended_to_quantum(self) -> 'StateSpace':
        """
        Extend SISO state-space to quantum MIMO state space in doubled-up ordering (see [#quantum]_).
        Returns extended `StateSpace`. Does not modify original.
        """
        from sympy import ImmutableMatrix, Matrix

        quantum_ss = deepcopy(self)
        quantum_ss.quantum = True
        quantum_ss.a = ImmutableMatrix(Matrix.diag(self.a, self.a.C))
        quantum_ss.b = ImmutableMatrix(Matrix.diag(self.b, self.b.C))
        quantum_ss.c = ImmutableMatrix(Matrix.diag(self.c, self.c.C))
        quantum_ss.d = ImmutableMatrix(Matrix.diag(self.d, self.d.C))
        quantum_ss.paired_operator_form = False
        return quantum_ss

    def reorder_to_paired_form(self) -> 'StateSpace':
        r"""
        Return a new StateSpace with the system matrices reordered so that the state vectors, inputs, and outputs are
        converted from doubled-up form,

        .. math::
            (a_1, a_2, \dots, a_n; a_1^\dagger, a_2^\dagger, \dots, a_n^\dagger)^T,

        to `paired operator form`,

        .. math::
            (a_1, a_1^\dagger; a_2, a_2^\dagger; \dots; a_n, a_n^\dagger)^T,

        Does nothing if self.paired_operator_form is True

        :return: StateSpace in paired up form.
        """
        if self.paired_operator_form:
            return self

        n = self.num_degrees_of_freedom
        if n % 2 != 0:
            raise DimensionError("num_degrees_of_freedom should be even for a quantum system")

        # construct the transformation matrix that reorders the elements
        # e.g. (1, 2, 3, 11, 22, 33) -> (1, 11, 2, 22, 3, 33)
        u = construct_permutation_matrix(n)

        # construct the matrices for the inputs and outputs
        u_i = construct_permutation_matrix(self.num_inputs)
        u_o = construct_permutation_matrix(self.num_outputs)

        # apply transformation and return
        return StateSpace(u*self.a*u.inv(), u*self.b*u_i.inv(), u_o*self.c*u.inv(), u_o*self.d*u_i.inv(),
                          paired_operator_form=True)

    def find_transformation_to_physically_realisable(self) -> sympy.Matrix:
        """
        Return the :math:`T` matrix that transforms the state space into a physically realisable one.

        Raise `StateSpaceError` if system is not possible to physically realise.

        Raise `ResultError` if there was some other unexpected error during finding T.
        """
        from sympy import Matrix, MatrixSymbol

        n = self.num_degrees_of_freedom
        if n % 2 != 0:
            raise DimensionError("num_degrees_of_freedom should be even for a quantum system")

        if config.params['checks']:
            self.raise_error_if_not_possible_to_realise()

        if config.params['checks'] and self.is_physically_realisable:
            return Matrix.eye(self.num_degrees_of_freedom)
        a, b, c, d = self

        # solve for x in both physical realisability equations
        j = j_matrix(self.num_degrees_of_freedom)
        j_i = j_matrix(self.num_inputs)
        x = MatrixSymbol('X', *j.shape)
        sol = solve_matrix_eqn([a * x + x * a.H + b * j_i * b.H, x * c.H + b * j_i * d.H], x)

        if len(sol) == 0:
            raise ResultError("Found no solution to the physical realisability equations.")

        x = sol[0]
        if not x.is_hermitian:
            raise ResultError("X must be Hermitian.")

        # diagonalise into X = P D P^\dagger where P is unitary via spectral theorem
        # i-th column of P is the i-th orthogonal eigenvector of X
        # i-th element of diagonal D is i-th real eigenvalue of X
        p, d = x.diagonalize(normalize=True)

        if config.params['checks']:
            assert simplify(p.H - p**-1, Matrix.zeros(*p.shape)), "p should be unitary"
            assert d.is_diagonal(), "d should be diagonal"
        eigenvals = list(d.diagonal())

        # need to re-arrange to form X = T J T^\dagger

        # first check that there are equal number of positive and negative eigenvalues, only then can be reordered
        # to match J
        def is_even(v):
            return v % 2 == 0

        def is_odd(v):
            return v % 2 != 0

        def is_pos(v):
            return v > 0

        def is_neg(v):
            return v < 0

        if len(list(filter(lambda v: v > 0, eigenvals))) != len(list(filter(lambda v: v < 0, eigenvals))):
            raise ResultError("Need equal number of positive and negative eigenvalues.")

        # find positive and negative eigenvalues that are in the wrong positions in the diagonal matrix
        from enum import Enum

        class Sign(Enum):
            POSITIVE = 1
            NEGATIVE = 2

        # return list of indices for eigenvalues in the wrong position for the given sign
        def get_indices_of_evs_in_wrong_positions_for_sign(sign=Sign.POSITIVE):
            return list(map(lambda v: v[0],
                            filter(lambda v: is_odd(v[0]) if sign == Sign.POSITIVE else is_even(v[0]),
                                   filter(lambda v: is_pos(v[1]) if sign == Sign.POSITIVE else is_neg(v[1]),
                                          enumerate(eigenvals)))))

        wrong_positive_indices = get_indices_of_evs_in_wrong_positions_for_sign(Sign.POSITIVE)
        wrong_negative_indices = get_indices_of_evs_in_wrong_positions_for_sign(Sign.NEGATIVE)
        indices_to_swap = zip(wrong_positive_indices, wrong_negative_indices)

        for wrong_pos_index, wrong_neg_index in indices_to_swap:
            # swap the eigenvalues
            eigenvals[wrong_pos_index], eigenvals[wrong_neg_index] = \
                eigenvals[wrong_neg_index], eigenvals[wrong_pos_index]
            # swap the eigenvectors
            p.col_swap(wrong_pos_index, wrong_neg_index)

        assert (len(get_indices_of_evs_in_wrong_positions_for_sign(Sign.POSITIVE)) == 0 and
                len(get_indices_of_evs_in_wrong_positions_for_sign(Sign.NEGATIVE)) == 0), \
            "Eigenvalues still not in order!"

        # factor out the square root of each eigenvalue into the eigenvectors
        from sympy import sqrt, radsimp
        scaled_evs = []
        for i, ev in enumerate(eigenvals):
            scale = abs(ev)
            scaled_evs.append(radsimp(ev / scale))
            p[:, i] *= sqrt(scale)

        if config.params['checks']:
            assert Matrix.diag(scaled_evs) == j, "Scaled eigenvalues matrix not recovered!"
            assert simplify(p * j * p.H - x, Matrix.zeros(*x.shape)), "Result not recovered as expected!"
        return simplify(p)

    def to_physically_realisable(self) -> 'StateSpace':
        """
        Return copy of state space transformed to a physically realisable state-space, or just return ``self`` if
        already physically realisable.

        Transforms to `paired operator form` first if needed.

        Raise `DimensionError` if system does not have even number of degrees of freedom.
        """
        n = self.num_degrees_of_freedom
        if n % 2 != 0:
            raise DimensionError("num_degrees_of_freedom should be even for a quantum system")

        if config.params['checks']:
            if self.is_physically_realisable:
                return self

        # transform to paired operator form is needed
        ss = self.reorder_to_paired_form()
        a, b, c, d = ss
        t = ss.find_transformation_to_physically_realisable()

        # apply transformation
        a = simplify(t**-1 * a * t)
        b = simplify(t**-1 * b)
        c = simplify(c * t)
        d = simplify(d)
        ss = StateSpace(a, b, c, d, paired_operator_form=True)

        if config.params['checks']:
            assert ss.is_physically_realisable, "Result was not physically realisable!"
        return ss

    @property
    def is_physically_realisable(self) -> bool:
        r"""
        Test physical realisability conditions using the `paired operator form` of the state-space.

        .. math::
            AJ + JA^\dagger + BJB^\dagger &= 0, \\
            JC^\dagger + BJD^\dagger &= 0.
        """
        from sympy import Matrix

        # reorder to paired operator form only if needed
        ss = self if self.paired_operator_form else self.reorder_to_paired_form()
        j = j_matrix(ss.num_degrees_of_freedom)
        j_i = j_matrix(ss.num_inputs)

        realisability1 = ss.a * j + j * ss.a.H + ss.b * j_i * ss.b.H
        realisability2 = j * ss.c.H + ss.b * j_i * ss.d.H

        cond1 = simplify(realisability1, Matrix.zeros(*realisability1.shape))
        cond2 = simplify(realisability2, Matrix.zeros(*realisability2.shape))
        return cond1 and cond2

    def raise_error_if_not_possible_to_realise(self) -> None:
        r"""
        Raises `StateSpaceError` if system cannot be physically realised according to the conditions given in
        `is_transfer_matrix_physically_realisable`.

        *Note* this does not imply that the system matrices :math:`(A, B, C, D)` are physically realisable, just that
        they can be transformed to a physically realisable form.
        """
        from sympy import Symbol, Matrix

        g = self.to_transfer_function()
        j = j_matrix(g.shape[0])
        s = Symbol('s')

        if not config.params['wolframscript']:
            from sympy import conjugate
            cond1 = simplify(g.subs(s, conjugate(s)).H * j * g.subs(s, -s) - j)

            import sympy.printing.mathematica as m
            print(m.mathematica_code(cond1[0, 0]))

            if cond1 != Matrix.zeros(*j.shape):
                raise StateSpaceError(f"Not possible to realise: {cond1} != 0")

            j = j_matrix(self.d.shape[1])
            cond2 = simplify(self.d * j * self.d.H - j)
            if cond2 != Matrix.zeros(*j.shape):
                raise StateSpaceError(f"Not possible to realise: {cond2} != 0")
        else:
            # HACK: We have to handle wolframscript a bit specially
            from sympy import conjugate
            cond1 = simplify(g.subs(s, conjugate(s)).H * j * g.subs(s, -s) - j, Matrix.zeros(*j.shape)) == b'True\n'
            if not cond1:
                raise StateSpaceError(f"Not possible to realise")

            j = j_matrix(self.d.shape[1])
            cond2 = simplify(self.d * j * self.d.H - j, Matrix.zeros(*j.shape))  == b'True\n'
            if not cond2:
                raise StateSpaceError(f"Not possible to realise")

    def to_skr(self) -> 'SKR':
        """
        Convert state space to SLH form as discussed in [synthesis]_, specifically returning the matrices
        :math:`(S, K, R)`.
        Assume physically realisable but won't error if it's not.
        """
        from sympy import I, Rational, BlockMatrix, Matrix
        j = j_matrix(self.num_degrees_of_freedom)
        ss = self.reorder_to_paired_form()

        # permutation matrix from (a_1, a_1^d; ...; a_n, a_n^d) -> (a_1, ..., a_n; a_1^d, ..., a_n^d)
        permutation_matrix = construct_permutation_matrix(ss.num_inputs) ** -1
        identity_block = Matrix(BlockMatrix([Matrix.eye(ss.num_inputs // 2), Matrix.zeros(ss.num_inputs // 2)]))

        # TODO: d should be different here, see my paper
        return SKR(ss.d, identity_block * permutation_matrix * ss.c, I * Rational(1, 4) * (j * ss.a - ss.a.H * j))

    def to_slh(self, symbol='a') -> 'SLH':
        """Create `StateSpace.to_skr` returning a `SLH` object using given symbol name ``symbol``, defaulting to 'a'."""
        s, k, r = self.to_skr()
        x0 = make_complex_ladder_state(r.shape[0] // 2, symbol)
        return SLH(s, k, r, x0)

    @property
    def num_degrees_of_freedom(self) -> int:
        """Returns num degrees of freedom if classical, or 2 * num degrees of freedom if quantum."""
        return self.a.shape[0]

    @property
    def num_inputs(self) -> int:
        """Returns num inputs if classical, or 2 * num inputs if quantum."""
        return self.b.shape[1]

    @property
    def num_outputs(self) -> int:
        """Returns num outputs if classical, or 2 * num outputs if quantum."""
        return self.c.shape[0]

    def pprint(self) -> None:
        """Pretty print the system matrices for debug or interactive programming purposes."""
        print(self)

    def _repr_latex_(self) -> str:
        """Display `StateSpace` in Jupyter notebook as LaTeX."""
        from sympy.printing.latex import latex
        lb = r"\\" if self.num_degrees_of_freedom > 4 else r",\,"  # only break lines if matrices are large
        return f"$$\\displaystyle A={latex(self.a)}{lb}B={latex(self.b)}{lb}C={latex(self.c)}{lb}D={latex(self.d)}$$"

    def __iter__(self) -> iter:
        """Returns iterator holding tuple with the four system matrices. Use for unpacking."""
        return iter((self.a, self.b, self.c, self.d))

    def __eq__(self, other) -> bool:
        """Equality for state spaces means that all the ABCD matrices are equal and both are or aren't quantum."""
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d

    def __str__(self) -> str:
        """Prettify the equation."""
        from sympy.printing.pretty import pretty
        return f"{pretty(self.a)}\n{pretty(self.b)}\n{pretty(self.c)}\n{pretty(self.d)}\n"

    def __repr__(self) -> str:
        return f"{repr(self.a)}\n{repr(self.b)}\n{repr(self.c)}\n{repr(self.d)}\n"

    def __hash__(self) -> int:
        return hash(tuple(self))


def j_matrix(num_dof: int) -> sympy.Matrix:
    r"""
    Return quantum :math:`J` matrix for a `paired operator form` `StateSpace` with given ``num_dof``,

    .. math::
        J = \text{diag}(1, -1;\dots;1, -1) \in \mathbb{R}^{2n\times 2n}.

    Raises ``ValueError`` if num_dof is not even.
    """
    if num_dof % 2 != 0:
        raise DimensionError("num_dof should be even for a quantum system")
    return sympy.Matrix.diag([1, -1] * (num_dof // 2))


def transfer_function_to_coeffs(expr: sympy.Expr, flip_s=True) -> 'Coefficients':
    """
    Extract transfer function coefficients from the given expression which is a function of frequency :math:`s` and a
    ratio of two polynomials.

    Returns a namedtuple containing two lists of length n and n-1 for the numerator and denominator respectively
    in *order of ascending powers*, where n is the order of expr in :math:`s`.

    The numerator list will be padded if the denominator is higher order than the numerator.

    A ``NotImplementedError`` will be raised if the denominator is lower order than the numerator.

    The coefficients are normalised such that the coefficient of the highest order term in the denominator is one.

    :param expr: ratio of two polynomials in :math:`s`
    :param flip_s: if True, set s to -s, that is use the Laplace convention defined in [#laplace]_
    :return: `Coefficients` instance
    """
    from sympy import Symbol, fraction
    s = Symbol('s')
    if flip_s:
        expr = expr.subs(s, -s)  # transform to our Laplace convention

    numers_and_denoms = tuple(map(lambda e: list(reversed(e.as_poly(s).all_coeffs())), fraction(expr)))
    numer_coeffs, denom_coeffs = numers_and_denoms

    if len(numer_coeffs) > len(denom_coeffs):
        raise NotImplementedError("TODO: currently denominator can't be lower order than numerator")

    # normalize w.r.t. highest order coeff in denominator
    highest_order_denom = denom_coeffs[-1]

    numer_coeffs, denom_coeffs = map(lambda coeffs: [n / highest_order_denom for n in coeffs], numers_and_denoms)

    # pad numer_coeffs with zeros at end until length of denom_coeffs
    numer_coeffs.extend([0] * (len(denom_coeffs) - len(numer_coeffs)))

    assert denom_coeffs[-1] == 1, "sanity check on normalisation failed"
    denom_coeffs = denom_coeffs[:-1]  # drop last element of denoms
    assert len(denom_coeffs) == len(numer_coeffs) - 1, "sanity check on lengths failed"

    return Coefficients(numer=numer_coeffs, denom=denom_coeffs)


class Coefficients(NamedTuple):
    """Represents the transfer function coefficients as returned by `transfer_function_to_coeffs`"""
    numer: List
    denom: List


"""SLH formalism"""


def concat(a: 'SLH', b: 'SLH') -> 'SLH':
    r"""
    Concatenate two `SLH` systems using the concatenation product. [synthesis]_

    Let :math:`G_1 = (S_1, K_1 x_{1,0}, \frac{1}{2} x_{1,0}^\dag R_2 x_{1,0}`) and
    :math:`G_2 = (S_2, K_2 x_{2,0}, \frac{1}{2} x_{2,0}^\dag R_2 x_{2,0})`, where :math:`x_{k,0} \equiv x_k(t=0)`.
    Then the concatenation product is defined as,

    .. math::
        G_1 \boxplus G_2 = \left(S_{1\boxplus2}, (K_1x_{1,0},K_2x_{2,0})^T,
        \frac{1}{2} x_{1,0}^\dag R_2 x_{1,0}+\frac{1}{2} x_{2,0}^\dag R_2 x_{2,0}\right),

    where,

    .. math::
        S_{1\boxplus2} = \begin{bmatrix}S_1 & 0 \\ 0 & S_2\end{bmatrix}.

    *For simplicity we assume the system's share no degrees of freedom*, i.e. :math:`x_{1,0} \neq x_{2,0}`.

    :param a: `SLH` object representing generalised open oscillator :math:`G_1`
    :param b: `SLH` object representing generalised open oscillator :math:`G_2`
    :return: `SLH` object representing concatenation of both
    """
    from sympy import Matrix
    s = Matrix.diag(a.s, b.s)
    k = Matrix.diag(a.k, b.k)
    r = Matrix.diag(a.r, b.r)
    x0 = Matrix([a.x0, b.x0])
    return SLH(s, k, r, x0)


def series(g_to, g_from):
    r"""
    *Not yet implemented*

    Series product representing the feeding of the output of ``g_from`` into the input of ``g_to``. The arguments
    are in this order to match the notation below. The generalised open oscillators :math:`G_1` and :math:`G_2` are as
    defined in `concat`.

    The series product is then defined as,

    .. math::
        G_2 \triangleleft G_1 = \bigg(&S_2S_1, K_2x_{2,0} + S_2 K_1x_{1,0}, \\
        &\frac{1}{2} x_{1,0}^\dag R_1 x_{1,0} + \frac{1}{2} x_{2,0}^\dag R_2 x_{2,0}
        + \frac{1}{2i} x_{2,0}^\dag (K_2^\dag S_2 K_1 - K_2^T S_2^* K_1^*) x_{1,0}\bigg).

    :param g_from: `SLH` object representing generalised open oscillator :math:`G_1`
    :param g_to: `SLH` object representing generalised open oscillator :math:`G_2`
    :return: `SLH` object representing concatenation of both
    """
    raise NotImplementedError("series(g_to, g_from) is not yet implemented.")


class SKR(NamedTuple):
    """Represents SKR matrices as returned by `StateSpace.to_skr`"""
    s: sympy.Matrix
    k: sympy.Matrix
    r: sympy.Matrix


class SLH:
    """
    Represents a generalised open oscillator in the SLH formalism. [synthesis]_

    Attributes: (all stored as sympy ``ImmutableMatrix`` objects)
        - ``s``: scattering matrix
        - ``k``: linear coupling matrix
        - ``r``: internal system Hamiltonian matrix
        - ``x0``: vector of system state Symbols
    """

    def __init__(self, s, k, r, x0=None):
        """Construct generalised open oscillator :math:`G = (S, L, H)`."""
        from sympy import ImmutableMatrix
        self.s = ImmutableMatrix(s)
        self.k = ImmutableMatrix(k)
        self.r = ImmutableMatrix(r)
        if x0 is None:
            x0 = make_complex_ladder_state(r.shape[0] // 2, 'a')
        self.x0 = ImmutableMatrix(x0)

    def _repr_latex_(self) -> str:
        """Display `SLH` in Jupyter notebook as LaTeX."""
        from sympy.printing.latex import latex

        # check if S is identity, if so display identity matrix instead
        n = self.s.shape[0]
        if self.s == sympy.Matrix.eye(n):
            s_latex_string = r"I_{%d\times%d}" % (n, n)
        else:
            s_latex_string = latex(self.s)

        x0d = latex(self.x0.H)
        x0 = latex(self.x0)
        return r"$$\displaystyle \left(%s, %s %s, %s %s %s\right)$$"\
               % (s_latex_string, latex(self.k), x0, x0d, latex(self.r), x0)

    def __repr__(self) -> str:
        return f"(S = {repr(self.s)}, K = {repr(self.k)}, R = {repr(self.r)})"

    def split(self) -> 'SplitNetwork':
        """Returns `split_system(self)`."""
        return split_system(self)

    @property
    def interaction_hamiltonian(self):
        """
        Returns the interaction Hamiltonian for the system via
        `interaction_hamiltonian_from_linear_coupling_operator(self.k * self.x0)`
        """
        return interaction_hamiltonian_from_linear_coupling_operator(self.k * self.x0)


def interaction_hamiltonian_from_k_matrix(k_matrix: sympy.Matrix) -> sympy.Expr:
    return interaction_hamiltonian_from_linear_coupling_operator(linear_coupling_operator_from_k_matrix(k_matrix))


def interaction_hamiltonian_from_linear_coupling_operator(l_operator: sympy.Matrix) -> sympy.Expr:
    r"""
    Calculate the idealised interaction hamiltonian for the given linear coupling operator.

    .. math::
        H_\text{int} = i[L^\dagger\ -L^\dagger] u,

    where :math:`u = (u_1, u_1^\dagger; \dots; u_m, u_m^\dagger)^T` and :math:`L \in \mathbb{L}^{m\times1}`

    Raises `DimensionError` if not ``l_operator`` not a column vector.
    """
    if l_operator.shape[1] != 1:
        raise DimensionError(f"L is not a column vector: {l_operator.shape}")

    from sympy import I, simplify, BlockMatrix
    states = make_complex_ladder_state(l_operator.shape[1], "u")
    h_int = I * sympy.Matrix(BlockMatrix([[l_operator.H, -l_operator.T]])) * states
    if h_int.shape != (1, 1):
        raise DimensionError(f"Expected interaction Hamiltonian to be scalar, instead: {h_int.shape}")
    return simplify(h_int[0, 0])


def linear_coupling_operator_from_k_matrix(k_matrix: sympy.Matrix, symbol: str = 'a') -> sympy.Matrix:
    r"""
    Calculate symbolic linear coupling operator from K matrix assuming `paired operator form`.

    .. math::
        L = K x_0,

    where :math:`x_0 = (a_1, a_1^\dagger; \dots; a_n, a_n^\dagger)^T`.

    ``symbol`` is the symbol name to use for the state variables.

    Raises `DimensionError` if not an even number of dimensions.
    """
    if k_matrix.shape[0] % 2 != 0 or k_matrix.shape[1] % 2 != 0:
        raise DimensionError(f"k_matrix does not have even dimensions: {k_matrix.shape}")

    states = make_complex_ladder_state(k_matrix.shape[1] // 2, symbol)
    return k_matrix * states


def hamiltonian_from_r_matrix(r_matrix: sympy.Matrix, symbol: str = 'a') -> sympy.Expr:
    r"""
    Calculate symbolic internal Hamiltonian from R matrix assuming `paired operator form`.

    .. math::
        H = x_0^\dagger R x_0,

    where :math:`x_0 = (a_1, a_1^\dagger; \dots; a_n, a_n^\dagger)` and :math:`R \in \mathbb{R}^{2n\times2n}`.

    ``symbol`` is the symbol name to use for the state variables.

    Raises `DimensionError` if not an even number of dimensions or if not square.
    """
    if not r_matrix.is_square:
        raise DimensionError(f"R is not square: {r_matrix.shape}")
    if r_matrix.shape[0] % 2 != 0:
        raise DimensionError(f"R does not have even number of rows/columns: {r_matrix.shape}")

    states = make_complex_ladder_state(r_matrix.shape[0] // 2, symbol)
    hamiltonian = states.H * r_matrix * states
    if hamiltonian.shape != (1, 1):
        raise DimensionError(f"Expected Hamiltonian to be a scalar, instead: {hamiltonian.shape}")
    return hamiltonian[0, 0]


def make_complex_ladder_state(num_dofs: int, symbol: str = 'a') -> sympy.Matrix:
    r"""
    Return matrix of complex ladder operators with ``2 * num_dofs`` elements.

    Use ``symbol`` keyword arg to set alternative symbol for variables instead of using ``a``

    For example, for ``num_dofs == 2``, result is :math:`(a_1, a_1^\dagger; a_2, a_2^\dagger)^T`.

    If ``num_dofs == 1`` then just returns :math:`(a, a^\dagger)^T`
    """
    from sympy import Symbol, Matrix
    states = []

    if num_dofs == 1:
        s = Symbol(symbol, commutative=False)
        states.append(s)
        states.append(s.conjugate())
    else:
        for i in range(num_dofs):
            s = Symbol(f"{symbol}_{i + 1}", commutative=False)
            states.append(s)
            states.append(s.conjugate())

    return Matrix(states)


"""Network Synthesis"""


class States:
    """Represents a collection of state variables that can be queried to extract different variables."""
    def __init__(self, states: sympy.Matrix):
        self.states = states

    def get_symbol(self, name: str) -> Optional[sympy.Symbol]:
        r"""
        Get symbol from self.states with given name.
        To get a conjugated variable, e.g. :math:`a_1^\dagger`, use ``conjugate(a_1)``.
        To get a primed variable write ``a'_1`` rather than ``a_1'``.
        """
        if not isinstance(name, str):
            raise TypeError("name must be a string")

        for symbol in self.states:
            if str(symbol) == name:
                return symbol

        return None

    def get_symbols(self, names: List[str]) -> List[sympy.Symbol]:
        """Same as `get_symbol` but takes and returns a list."""
        return list(map(self.get_symbol, names))

    def index_of(self, symbol: sympy.Symbol) -> Optional[int]:
        """Return index of symbol in ``self.states``, or ``None`` if not found."""
        return list(self.states).index(symbol)

    def _repr_latex_(self) -> str:
        """Display `StateSpace` in Jupyter notebook as LaTeX."""
        from sympy.printing.latex import latex
        return f"$$\\displaystyle {latex(self.states)}$$"


class SplitNetwork:
    """
    Represents a tuple of ([G_1, ..., G_n], H^d), as returned by `split_system`.

    Each :math:`G_i` is assumed to be series fed into :math:`G_{i+1}`, however if :math:`L = 0` for :math:`G_i` it
    can be considered as not being series connected as the external field will only couple to the auxiliary degree of
    freedom which is adiabatically eliminated.
    """

    def __init__(self, gs: List[SLH], h_d: sympy.Matrix):
        self.gs = gs
        self.h_d = h_d

    def __getitem__(self, item):
        if item == 0:
            return self.gs
        elif item == 1:
            return self.h_d
        else:
            raise IndexError("Out of range (index must be 0 or 1)")

    def __iter__(self):
        return iter((self.gs, self.h_d))

    def __str__(self):
        return str(tuple(self))

    @property
    def states(self) -> sympy.Matrix:
        """Return the state symbols for the main and auxiliary mode for the entire network."""
        states = make_complex_ladder_state(len(self.gs), 'a')
        aux_states = make_complex_ladder_state(len(self.gs), 'a\'')
        return sympy.Matrix(sympy.BlockMatrix([[states], [aux_states]]))

    @property
    def input_output_symbols(self) -> sympy.Matrix:
        """Return the symbols for the input and output fields."""
        # special case for one degree of freedom
        if len(self.gs) == 1:
            ain, aind = make_complex_ladder_state(1, f'ain')
            aout, aoutd = make_complex_ladder_state(1, f'aout')
            return sympy.Matrix([ain, aind, aout, aoutd])
        else:
            symbols = []

            for i in range(len(self.gs)):
                ain, aind = make_complex_ladder_state(1, f'ain_{i + 1}')
                aout, aoutd = make_complex_ladder_state(1, f'aout_{i + 1}')
                symbols.extend([ain, aind, aout, aoutd])

            return sympy.Matrix(symbols)

    class InteractionHamiltonian:
        r"""
        Represents the interaction Hamiltonian as a Matrix in order

        .. math::
            (a_1, a_1^\dagger, \dots, a_n, a_n^\dagger; a_1', a_1'^\dagger,\dots, a_n', a_n'^\dagger),

        where :math:`a_i` is the annihilation operator for the main cavity mode of the i-th system and
        :math:`a_i'` is for the corresponding auxiliary mode.
        """

        def __init__(self, h: sympy.Matrix):
            self.h = h

        @property
        def expr(self) -> sympy.Expr:
            x = self.states
            return (x.H * self.h * x)[0, 0]

        @property
        def states(self) -> sympy.Matrix:
            """Return the state symbols for the main and auxiliary modes for the network."""
            states = make_complex_ladder_state(self.h.rows // 4, 'a')
            aux_states = make_complex_ladder_state(self.h.rows // 4, 'a\'')
            return sympy.Matrix(sympy.BlockMatrix([[states], [aux_states]]))

        @property
        def dynamical_matrix(self) -> sympy.Matrix:
            """Compute dynamical matrix as shown in ``notes/eqns-of-motion-from-hamiltonian-matrix.pdf``."""
            from sympy import Matrix, I
            n = self.h.rows // 2
            j_1 = Matrix([[0, 1], [-1, 0]])
            j_mat = Matrix(sympy.BlockDiagMatrix(*([j_1] * n)))
            k_mat = Matrix.diag([-1, 1] * n)

            # row swapping matrix
            theta_1 = Matrix([[0, 1], [1, 0]])
            theta_mat = Matrix(sympy.BlockDiagMatrix(*([theta_1] * n)))

            a_mat = I * (j_mat.T * self.h.T * theta_mat + k_mat * self.h)

            if a_mat.shape != (2*n, 2*n):
                raise DimensionError(f"Expected a_mat to have shape {(2*n, 2*n)}, instead had {a_mat.shape}")
            return a_mat

        @property
        def equations_of_motion(self) -> sympy.Matrix:
            r"""
            Compute the Heisenberg equations of motion of the interacting terms (not including the Langevin equations)
            , using the Bosonic commutation relations

            .. math::
                [a_i, a_j] = 0, \quad [a_i, a_j^\dagger] = \delta_{i, j},

            and

            .. math::
                [a_i', a_j'] = 0, \quad [a_i', a_j'^\dagger] = \delta_{i, j}

            Returns eqns in order
            :math:`(a_1, a_1^\dagger, \dots, a_n, a_n^\dagger; a_1', a_1'^\dagger,\dots, a_n', a_n'^\dagger)`.

            :returns column vector of equations of motion
            """
            x = self.states
            x_dot = self.dynamical_matrix * x
            if x_dot.shape != (x.rows, 1):
                raise DimensionError(f"Expected x_dot to have shape ({x.rows}, 1), instead had {x_dot.shape}")
            return x_dot

    @property
    def interaction_hamiltonian(self) -> InteractionHamiltonian:
        """
        Compute the interaction Hamiltonian between internal degrees of freedom in the network, not including
        the external continuum fields.
        """
        from sympy import BlockDiagMatrix, sqrt, conjugate, Matrix, Symbol, I

        h_int = Matrix(BlockDiagMatrix(self.h_d, Matrix.zeros(*self.h_d.shape)))

        offset = self.h_d.rows

        # special case for one internal degree of freedom
        if len(self.gs) == 1:
            gamma = Symbol(f'gamma', positive=True, real=True)
            alpha, beta = self.gs[0].k
            epsilon_1 = beta * sqrt(2 * gamma)
            epsilon_2 = -conjugate(alpha * sqrt(2 * gamma))

            h_int[1, offset] = -I / 2 * conjugate(epsilon_1)
            h_int[0, offset + 1] = I / 2 * epsilon_1
            h_int[1, offset + 1] = -I / 2 * conjugate(epsilon_2)
            h_int[0, offset] = I / 2 * epsilon_2
        else:
            for i, g in enumerate(self.gs):
                gamma_i = Symbol(f'gamma_{i + 1}', positive=True, real=True)
                alpha, beta = g.k
                epsilon_1 = beta * sqrt(2 * gamma_i)
                epsilon_2 = -conjugate(alpha * sqrt(2 * gamma_i))

                # a <-> a'
                h_int[i * 2 + 1, offset + i * 2] = -I / 2 * conjugate(epsilon_1)
                # a^\dag <-> a'^\dag
                h_int[i * 2, offset + i * 2 + 1] = I / 2 * epsilon_1
                # a <-> a'^\dag
                h_int[i * 2 + 1, offset + i * 2 + 1] = -I / 2 * conjugate(epsilon_2)
                # a^\dag <-> a'
                h_int[i * 2, offset + i * 2] = I / 2 * epsilon_2

        return SplitNetwork.InteractionHamiltonian(h_int)

    @property
    def input_output_eqns(self) -> sympy.Matrix:
        """Calculate the input-output equations for the auxiliary fields as column vector."""
        from sympy import Matrix, Symbol

        eqns = []

        # special case for one internal degree of freedom
        if len(self.gs) == 1:
            ap, apd = make_complex_ladder_state(1, f'a\'')
            ain, aind = make_complex_ladder_state(1, f'ain')
            aout, aoutd = make_complex_ladder_state(1, f'aout')
            gamma = Symbol(f'gamma', positive=True, real=True)
            eqns.append(-aout + ain - sympy.sqrt(2 * gamma) * ap)
            eqns.append(-aoutd + aind - sympy.sqrt(2 * gamma) * apd)
        else:
            for i, g in enumerate(self.gs):
                ap, apd = make_complex_ladder_state(1, f'a\'_{i + 1}')
                ain, aind = make_complex_ladder_state(1, f'ain_{i + 1}')
                aout, aoutd = make_complex_ladder_state(1, f'aout_{i + 1}')
                gamma_i = Symbol(f'gamma_{i + 1}', positive=True, real=True)

                # check if gamma_i should be zero (i.e. if there is no coupling)
                if g.k == Matrix.zeros(1, 2):
                    gamma_i = 0

                eqns.append(-aout + ain - sympy.sqrt(2 * gamma_i) * ap)
                eqns.append(-aoutd + aind - sympy.sqrt(2 * gamma_i) * apd)

        return Matrix(eqns)

    @property
    def state_vector(self) -> sympy.Matrix:
        r"""
        Returns column vector of states as follows:

        .. math::
            (a_1, a_1^\dagger, \dots, a_n, a_n^\dagger;
            a_1', a_1'^\dagger,\dots, a_n', a_n'^\dagger;
            a_\text{in 1}, a_\text{in 1}^\dagger, a_\text{in m},a_\text{in m}^\dagger;
            a_\text{out 1}, a_\text{out 1}^\dagger, a_\text{out m}, a_\text{out m}^\dagger)^T

        for system with ``n`` internal degrees of freedom and ``m`` inputs and outputs.
        """
        return sympy.Matrix(sympy.BlockMatrix([[self.states], [self.input_output_symbols]]))

    class DynamicalMatrix:
        """
        Represents the dynamical matrix as returned by `SplitNetwork.dynamical_matrix`.

        Used to calculate the transfer functions between any two quantities in the system.
        """

        def __init__(self, matrix: sympy.Matrix, states: Union[sympy.Matrix, States]):
            self.matrix = matrix
            self.states = states if isinstance(states, States) else States(states)

        class TransferMatrix:
            """
            Represents the results of `DynamicalMatrix.transfer_matrix`. Used to extract the transfer functions.
            """

            def __init__(self, matrix: sympy.Matrix, states: Union[sympy.Matrix, States]):
                self.matrix = matrix
                self.states = states if isinstance(states, States) else States(states)

            @staticmethod
            def _get_symbol_index(states: States, variable: Union[sympy.Symbol, str]) -> int:
                """
                If ``variable`` is a string, try to get it from ``states``.
                Then try to get the index of it within ``states``.
                """
                if isinstance(variable, str):
                    variable = states.get_symbol(variable)
                    if variable is None:
                        raise IndexError("String not found in self.states")

                index = states.index_of(variable)
                if index is None:
                    raise IndexError("Symbol not found in self.states")
                return index

            def closed_loop(self, excitation: Union[sympy.Symbol, str], variable: Union[sympy.Symbol, str])\
                    -> sympy.Expr:
                """
                Get the closed loop transfer function from the excitation to the variable.

                The variables can be given as strings (which are passed to `States.get_symbol`) or as sympy Symbols.

                E.g. ``closed_loop("ain", "aout")`` is the closed-loop transfer function from ``ain`` to ``aout``.
                """

                missing_symbols = []
                excitation_index = None
                variable_index = None

                try:
                    excitation_index = self._get_symbol_index(self.states, excitation)
                except IndexError:
                    missing_symbols.append(excitation)

                try:
                    variable_index = self._get_symbol_index(self.states, variable)
                except IndexError:
                    missing_symbols.append(variable)

                if len(missing_symbols) > 0:
                    raise IndexError("Symbols " + str(", ".join(missing_symbols)) + " missing from self.states")

                return self.matrix[variable_index, excitation_index]

            def closed_loop_gain(self, excitation: Union[sympy.Symbol, str]):
                r"""
                Calculate the closed loop gain from the excitation to the corresponding variable.

                I.e. if excitation is :math:`\hat{a}_\text{exc}` then returns the transfer function from
                :math:`\hat{a}_\text{exc}` to :math:`\hat{a}`.
                """
                try:
                    excitation_index = self._get_symbol_index(self.states, excitation)
                except IndexError:
                    raise IndexError("Symbol missing from self.states")

                return self.matrix[excitation_index, excitation_index]

            def open_loop(self, excitation: Union[sympy.Symbol, str], variable: Union[sympy.Symbol, str])\
                    -> sympy.Expr:
                """
                Calculate the open loop transfer function from the excitation to the variable (the closed loop transfer
                function divided by the closed loop gain).
                """
                return self.closed_loop(excitation, variable) / self.closed_loop_gain(excitation)

        @property
        def transfer_matrix(self) -> TransferMatrix:
            """Calculate the transfer matrix for the system by adding an excitation to each mode."""
            excitation_matrix = sympy.eye(*self.matrix.shape)
            return self.TransferMatrix((excitation_matrix - self.matrix) ** -1, self.states)

        @property
        def eqns(self) -> sympy.Matrix:
            """Calculate matrix of RHS of frequency-domain equations :math:`v = Mv`."""
            return self.matrix * self.states.states

    @property
    def dynamical_matrix(self) -> DynamicalMatrix:
        r"""
        Calculate the full frequency-domain dynamical matrix :math:`M` for the system, include input and output
        equations, such that :math:`v = M v` where :math:`v` is the full frequency-domain state vector in
        the order returned by `state_vector`.

        TODO: use a sparse matrix
        """
        from sympy import BlockMatrix, Matrix, Symbol, I
        s = Symbol('s')
        # A matrix part (internal modes -> internal modes)
        a_mat = self.interaction_hamiltonian.dynamical_matrix / (-s)

        # Add detuning and internal squeezing terms
        for i, g in enumerate(self.gs):
            a_mat[2*i, 2*i] = I * 2 * g.r[0, 0] / (-s)
            a_mat[2*i+1, 2*i+1] = -I * 2 * g.r[1, 1] / (-s)
            # TODO: check these
            a_mat[2*i+1, 2*i] = I * 2 * g.r[0, 1] / (-s)
            a_mat[2*i, 2*i+1] = -I * 2 * g.r[1, 0] / (-s)

        # Add B matrix part (inputs -> internal modes) and dissipation terms to a_mat
        n = len(self.gs)
        b_mat = Matrix.zeros(n*4, n*4)

        # special case for one internal degree of freedom
        if n == 1:
            gamma = Symbol(f'gamma', positive=True, real=True)
            a_mat[2, 2] = -gamma / (-s)
            a_mat[3, 3] = -gamma / (-s)
            b_mat[2, 0] = sympy.sqrt(2 * gamma) / (-s)
            b_mat[3, 1] = sympy.sqrt(2 * gamma) / (-s)
        else:
            for i, g in enumerate(self.gs):
                gamma_i = Symbol(f'gamma_{i + 1}', positive=True, real=True)

                # check if gamma_i should be zero (i.e. if there is no coupling)
                if g.k == Matrix.zeros(1, 2):
                    gamma_i = 0

                aux_mode_index = n * 2 + 2 * i  # row index of aux mode within a_mat
                # a' -> a'
                a_mat[aux_mode_index, aux_mode_index] = -gamma_i / (-s)
                a_mat[aux_mode_index+1, aux_mode_index+1] = -gamma_i / (-s)
                # ain -> a'
                b_mat[aux_mode_index, i] = sympy.sqrt(2 * gamma_i) / (-s)
                b_mat[aux_mode_index+1, i+1] = sympy.sqrt(2 * gamma_i) / (-s)

        # Add C matrix part (internal modes -> outputs)
        c_mat = Matrix.zeros(n*4, n*4)

        if n == 1:
            gamma = Symbol(f'gamma', positive=True, real=True)
            c_mat[2, 2] = -sympy.sqrt(2 * gamma)
            c_mat[3, 3] = -sympy.sqrt(2 * gamma)
        else:
            for i, g in enumerate(self.gs):
                gamma_i = Symbol(f'gamma_{i + 1}', positive=True, real=True)

                # check if gamma_i should be zero (i.e. if there is no coupling)
                if g.k == Matrix.zeros(1, 2):
                    gamma_i = 0

                aux_mode_index = n * 2 + 2 * i  # column index of aux mode within c_mat
                output_mode_index = aux_mode_index  # row index of output mode within c_mat
                c_mat[output_mode_index, aux_mode_index] = -sympy.sqrt(2 * gamma_i)
                c_mat[output_mode_index+1, aux_mode_index+1] = -sympy.sqrt(2 * gamma_i)

        # Add D matrix part (direct feed equations)
        d_mat = Matrix.zeros(n*4, n*4)

        if n == 1:
            d_mat[2, 0] = 1
            d_mat[3, 1] = 1
        else:
            for i, g in enumerate(self.gs):
                output_mode_index = n * 2 + 2 * i  # row index of output modes
                input_mode_index = i  # column index of inputs modes
                d_mat[output_mode_index, input_mode_index] = 1
                d_mat[output_mode_index+1, input_mode_index+1] = 1

        # add the series feed terms to the d matrix
        for i in range(n - 1):
            output_mode_index = n * 2 + 2 * i  # row index of output modes
            input_mode_index = i  # column index of inputs modes
            d_mat[input_mode_index + 2, output_mode_index] = 1
            d_mat[input_mode_index + 2 + 1, output_mode_index + 1] = 1

        m = Matrix(BlockMatrix([[a_mat, b_mat], [c_mat, d_mat]]))
        return SplitNetwork.DynamicalMatrix(m, self.state_vector)

    @property
    def transfer_matrix(self) -> DynamicalMatrix.TransferMatrix:
        """Shortcut for `self.dynamical_matrix.transfer_matrix`"""
        return self.dynamical_matrix.transfer_matrix

    @property
    def tfm(self) -> DynamicalMatrix.TransferMatrix:
        """Shortcut for `self.dynamical_matrix.transfer_matrix`"""
        return self.dynamical_matrix.transfer_matrix

    @property
    def aux_coupling_constants(self) -> List[sympy.Symbol]:
        """
        Return a list of the Sympy symbols use for the coupling constants for the auxiliary modes.

        Examples:
            >>> from sympy import symbols
            >>> s = symbols('s')
            >>> gamma_f, omega_s = symbols('gamma_f omega_s', real=True, positive=True)
            >>> network = tf2network((s**2 + s * gamma_f + omega_s**2) / (s**2 - s * gamma_f + omega_s**2))
            >>> gamma_1, gamma_2 = network.aux_coupling_constants
        """
        if len(self.gs) == 1:
            return [sympy.Symbol('gamma', positive=True, real=True)]
        else:
            return list(map(lambda i: sympy.Symbol(f'gamma_{i + 1}', positive=True, real=True), range(len(self.gs))))


def split_system(open_osc: SLH) -> SplitNetwork:
    """
    Split n degree of freedom open oscillator into n one degree of freedom open oscillators and a direct interaction
    Hamiltonian matrix.

    :param open_osc: the n degree of freedom open oscillator to split
    :return: a `SplitNetwork` which represents a tuple of ([G_1, ..., G_n], H^d)
    """
    from sympy import Matrix, I

    dof = open_osc.r.shape[0] // 2

    if dof == 1:  # already separated
        return SplitNetwork([open_osc], Matrix.zeros(2, 2))

    # get the list of 2x2 diagonal block matrices of R
    r_blocks = list(map(lambda i: open_osc.r[i:(i+2), i:(i+2)], range(0, 2 * dof, 2)))
    # get the list of mx2 k matrices
    k_blocks = list(map(lambda i: open_osc.k[0:2, i:(i+2)], range(0, 2 * dof, 2)))

    # constructs one open oscillator
    def construct_slh(index, k_r):
        k, r = k_r
        return SLH(Matrix.eye(2, 2), k, r, make_complex_ladder_state(1, f'a_{index + 1}'))

    # construct the open oscillators
    gs = list(map(lambda v: construct_slh(*v), enumerate(zip(k_blocks, r_blocks))))

    # TODO: for now we assume no scattering for simplicity

    h_d = Matrix.zeros(dof * 2, dof * 2)

    # construct Hamiltonian interaction matrix
    for j in range(0, dof - 1):
        for k in range(j + 1, dof):
            h_d[(j*2):((j+1)*2), (k*2):((k+1)*2)] = 2 * (open_osc.r[(k*2):((k+1)*2), (j*2):((j+1)*2)].H
                - 1 / (2 * I) * (k_blocks[k].H * k_blocks[j] - k_blocks[k].T * k_blocks[j].C))

    return SplitNetwork(gs, h_d)
