
from copy import deepcopy
from collections import namedtuple
from functools import lru_cache

from sympy import Matrix, Symbol, fraction, ImmutableMatrix, MatrixSymbol, I

from simba.utils import solve_matrix_eqn, construct_transformation_matrix, matrix_simplify
from simba.errors import DimensionError, CoefficientError, StateSpaceError, ResultError
import simba.config as config

"""State-spaces and transfer functions"""


def is_transfer_matrix_physically_realisable(expr, d_matrix=None):
    r"""
    Check if transfer matrix :math:`\mathbf{G}(s)` given by ``expr`` and direct-feed matrix ``d_matrix``
    is possible to physically realise by the conditions given in [transfer-function]_, i.e. that it obeys,

    .. math::
        \mathbf{G}^\sim(s) J \mathbf{G}(s) = J,\ \ D J D^\dagger = J,

    where :math:`\mathbf{G}^\sim(s) \equiv \mathbf{G}^\dag(-s^*)`.

    :param expr: a sympy Matrix representing the transfer Matrix :math:`\mathbf{G}(s)`
    :param d_matrix: a sympy Matrix representing the direct-feed Matrix, defaulting to the identity matrix
    """
    j = j_matrix(expr.shape[0])
    if not d_matrix:
        d_matrix = Matrix.eye(*j.shape)

    s = Symbol('s')
    from sympy import simplify, conjugate
    cond1 = simplify(expr.subs(s, -conjugate(s)).H * j * expr - j) == Matrix.zeros(*j.shape)
    cond2 = simplify(d_matrix * j * d_matrix.H - j) == Matrix.zeros(*j.shape)

    return cond1 and cond2


def transfer_func_coeffs_to_state_space(numer, denom):
    """See `StateSpace.from_transfer_function_coeffs`."""
    return StateSpace.from_transfer_function_coeffs(numer, denom)


def transfer_function_to_state_space(expr):
    """See `StateSpace.from_transfer_function`."""
    return StateSpace.from_transfer_function(expr)


def transfer_function_to_realisable_state_space(expr):
    """Convert given transfer function to physically realisable state space if possible."""
    return transfer_function_to_state_space(expr).to_physically_realisable()


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
    def from_transfer_function_coeffs(cls, numer, denom):
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
    def from_transfer_function(cls, expr):
        """Call `from_transfer_function_coeffs` passing the expression to `transfer_function_to_coeffs`."""
        return cls.from_transfer_function_coeffs(*transfer_function_to_coeffs(expr))

    def to_transfer_function(self):
        """Calculate transfer function matrix for the system using the convention given by [#laplace]_."""
        s = Symbol('s')
        return self.c * (-s * Matrix.eye(self.a.shape[0]) - self.a).inv() * self.b + self.d

    def extended_to_quantum(self):
        """
        Extend SISO state-space to quantum MIMO state space in doubled-up ordering (see [#quantum]_).
        Returns extended `StateSpace`. Does not modify original.
        """
        quantum_ss = deepcopy(self)

        quantum_ss.quantum = True
        quantum_ss.a = ImmutableMatrix(Matrix.diag(self.a, self.a.C))
        quantum_ss.b = ImmutableMatrix(Matrix.diag(self.b, self.b.C))
        quantum_ss.c = ImmutableMatrix(Matrix.diag(self.c, self.c.C))
        quantum_ss.d = ImmutableMatrix(Matrix.diag(self.d, self.d.C))
        return quantum_ss

    @lru_cache()
    def reorder_to_paired_form(self):
        r"""
        Return a new StateSpace with the system matrices reordered so that the state vectors, inputs, and outputs are
        converted from doubled-up form,

        .. math::
            (a_1, a_2, \dots, a_n; a_1^\dagger, a_2^\dagger, \dots, a_n^\dagger)^T,

        to `paired operator form`,

        .. math::
            (a_1, a_1^\dagger; a_2, a_2^\dagger; \dots; a_n, a_n^\dagger)^T,


        Result with be cached using ``functools.lru_cache``, so subsequent calls should be "free".

        :return: StateSpace in paired up form.
        """
        n = self.num_degrees_of_freedom
        if n % 2 != 0:
            raise DimensionError("num_degrees_of_freedom should be even for a quantum system")

        # construct the transformation matrix that reorders the elements
        # e.g. (1, 2, 3, 11, 22, 33) -> (1, 11, 2, 22, 3, 33)
        u = construct_transformation_matrix(n)

        # construct the matrices for the inputs and outputs
        u_i = construct_transformation_matrix(self.num_inputs)
        u_o = construct_transformation_matrix(self.num_outputs)

        # apply transformation and return
        return StateSpace(u*self.a*u.inv(), u*self.b*u_i.inv(), u_o*self.c*u.inv(), u_o*self.d*u_i.inv(),
                          paired_operator_form=True)

    def find_transformation_to_physically_realisable(self):
        """
        Return the :math:`T` matrix that transforms the state space into a physically realisable one.

        TODO: needs a bit of testing

        Raise `StateSpaceError` if system is not possible to physically realise.

        Raise `ResultError` if there was some other unexpected error during finding T.
        """
        n = self.num_degrees_of_freedom
        if n % 2 != 0:
            raise DimensionError("num_degrees_of_freedom should be even for a quantum system")

        self.raise_error_if_not_possible_to_realise()

        if config.params['checks'] and self.is_physically_realisable:
            return Matrix.eye(self.num_degrees_of_freedom)
        a, b, c, d = self

        # solve for x in both physical realisability equations
        j = j_matrix(self.num_degrees_of_freedom)
        j_i = j_matrix(self.num_inputs)
        x = MatrixSymbol('X', *j.shape)
        sol = solve_matrix_eqn([a * x + x * a.H + b * j_i * b.H, x * c.H + b * j_i * d.H], x)
        if len(sol) != 1:
            raise ResultError("Expected one and exactly one result.")

        x = sol[0]
        if not x.is_hermitian:
            raise ResultError("X must be Hermitian.")

        # diagonalise into X = P D P^\dagger where P is unitary via spectral theorem
        # i-th column of P is the i-th orthogonal eigenvector of X
        # i-th element of diagonal D is i-th real eigenvalue of X
        p, d = x.diagonalize(normalize=True)

        if config.params['checks']:
            assert matrix_simplify(p.H - p**-1) == Matrix.zeros(*p.shape), "p should be unitary"
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
            assert matrix_simplify(p * j * p.H - x) == Matrix.zeros(*x.shape), "Result not recovered as expected!"
        return matrix_simplify(p)

    def to_physically_realisable(self):
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
        a, b, c, d = self if self.paired_operator_form else self.reorder_to_paired_form()
        ss = StateSpace(a, b, c, d, paired_operator_form=True)
        t = ss.find_transformation_to_physically_realisable()

        from sympy import simplify

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
    def is_physically_realisable(self):
        r"""
        Test physical realisability conditions using the `paired operator form` of the state-space.

        .. math::
            AJ + JA^\dagger + BJB^\dagger &= 0, \\
            JC^\dagger + BJD^\dagger &= 0.
        """
        # reorder to paired operator form only if needed
        ss = self if self.paired_operator_form else self.reorder_to_paired_form()
        j = j_matrix(ss.num_degrees_of_freedom)
        j_i = j_matrix(ss.num_inputs)

        realisability1 = ss.a * j + j * ss.a.H + ss.b * j_i * ss.b.H
        realisability2 = j * ss.c.H + ss.b * j_i * ss.d.H

        cond1 = matrix_simplify(realisability1) == Matrix.zeros(*realisability1.shape)
        cond2 = matrix_simplify(realisability2) == Matrix.zeros(*realisability2.shape)
        return cond1 and cond2

    def raise_error_if_not_possible_to_realise(self):
        r"""
        Raises `StateSpaceError` if system cannot be physically realised according to the conditions given in
        `is_transfer_matrix_physically_realisable`.

        *Note* this does not imply that the system matrices :math:`(A, B, C, D)` are physically realisable, just that
        they can be transformed to a physically realisable form.
        """
        g = self.to_transfer_function()
        j = j_matrix(g.shape[0])
        s = Symbol('s')

        from sympy import simplify, conjugate
        cond1 = simplify(g.subs(s, -conjugate(s)).H * j * g - j)
        if cond1 != Matrix.zeros(*j.shape):
            raise StateSpaceError(f"Not possible to realise: {cond1} != 0")

        j = j_matrix(self.d.shape[1])
        cond2 = simplify(self.d * j * self.d.H - j)
        if cond2 != Matrix.zeros(*j.shape):
            raise StateSpaceError(f"Not possible to realise: {cond2} != 0")

    def to_skr(self):
        """
        Convert state space to SLH form as discussed in [synthesis]_, specifically returning the matrices
        :math:`(S, K, R)`.
        Assume physically realisable but won't error if it's not.
        """
        from sympy import I, Rational
        j = j_matrix(self.num_degrees_of_freedom)
        SKR = namedtuple('SKR', ['s', 'k', 'r'])
        return SKR(self.d, self.d**-1 * self.c, I * Rational(1, 4) * (j * self.a - self.a.H * j))

    def to_slh(self, symbol='a'):
        """Create `StateSpace.to_skr` returning a `SLH` object using given symbol name ``symbol``, defaulting to 'a'."""
        s, k, r = self.to_skr()
        x0 = make_complex_ladder_state(r.shape[0] // 2, symbol)
        return SLH(s, k, r, x0)

    @property
    def num_degrees_of_freedom(self):
        """Returns num degrees of freedom if classical, or 2 * num degrees of freedom if quantum."""
        return self.a.shape[0]

    @property
    def num_inputs(self):
        """Returns num inputs if classical, or 2 * num inputs if quantum."""
        return self.b.shape[1]

    @property
    def num_outputs(self):
        """Returns num outputs if classical, or 2 * num outputs if quantum."""
        return self.c.shape[0]

    def pprint(self):
        """Pretty print the system matrices for debug or interactive programming purposes."""
        print(self)

    def _repr_latex_(self):
        """Display `StateSpace` in Jupyter notebook as LaTeX."""
        from sympy.printing.latex import latex
        lb = r"\\" if self.num_degrees_of_freedom > 4 else r",\,"  # only break lines if matrices are large
        return f"$$\\displaystyle A={latex(self.a)}{lb}B={latex(self.b)}{lb}C={latex(self.c)}{lb}D={latex(self.d)}$$"

    def __iter__(self):
        """Returns iterator holding tuple with the four system matrices. Use for unpacking."""
        return iter((self.a, self.b, self.c, self.d))

    def __eq__(self, other):
        """Equality for state spaces means that all the ABCD matrices are equal and both are or aren't quantum."""
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d

    def __str__(self):
        """Prettify the equation."""
        from sympy.printing.pretty import pretty
        return f"{pretty(self.a)}\n{pretty(self.b)}\n{pretty(self.c)}\n{pretty(self.d)}\n"

    def __repr__(self):
        return f"{repr(self.a)}\n{repr(self.b)}\n{repr(self.c)}\n{repr(self.d)}\n"

    def __hash__(self):
        return hash(tuple(self))


def j_matrix(num_dof):
    r"""
    Return quantum :math:`J` matrix for a `paired operator form` `StateSpace` with given ``num_dof``,

    .. math::
        J = \text{diag}(1, -1;\dots;1, -1) \in \mathbb{R}^{2n\times 2n}.

    Raises ``ValueError`` if num_dof is not even.
    """
    if num_dof % 2 != 0:
        raise DimensionError("num_dof should be even for a quantum system")
    return Matrix.diag([1, -1] * (num_dof // 2))


def transfer_function_to_coeffs(expr, flip_s=True):
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
    :return: ``Coefficients`` namedtuple with fields: ``(numer=[b_n, ..., b_0], denom=[a_n, ..., a_1])``
    """
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

    Coefficients = namedtuple('Coefficients', ['numer', 'denom'])
    return Coefficients(numer=numer_coeffs, denom=denom_coeffs)


"""SLH formalism"""


def concat(a, b):
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
    s = Matrix.diag(a.s, b.s)
    k = Matrix.diag(a.k, b.k)
    r = Matrix.diag(a.r, b.r)
    x0 = Matrix([a.x0, b.x0])
    return SLH(s, k, r, x0)


def series(g_to, g_from):
    r"""
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


class SLH:
    """
    Represents a generalised open oscillator in the SLH formalism. [synthesis]_

    Attributes: (all stored as sympy ``ImmutableMatrix`` objects)
        - ``s``: scattering matrix
        - ``k``: linear coupling matrix
        - ``r``: internal system Hamiltonian matrix
        - ``x0``: vector of system state Symbols
    """

    def __init__(self, s, k, r, x0):
        """Construct generalised open oscillator :math:`G = (S, L, H)`."""
        self.s = ImmutableMatrix(s)
        self.k = ImmutableMatrix(k)
        self.r = ImmutableMatrix(r)
        self.x0 = ImmutableMatrix(x0)

    def _repr_latex_(self):
        """Display `SLH` in Jupyter notebook as LaTeX."""
        from sympy.printing.latex import latex

        # check if S is identity, if so display identity matrix instead
        n = self.s.shape[0]
        if self.s == Matrix.eye(n):
            s_latex_string = r"I_{%d\times%d}" % (n, n)
        else:
            s_latex_string = latex(self.s)

        x0d = latex(self.x0.H)
        x0 = latex(self.x0)
        return r"$$\displaystyle \left(%s, %s %s, \frac{1}{2} %s %s %s\right)$$"\
               % (s_latex_string, latex(self.k), x0, x0d, latex(self.r), x0)

    def __repr__(self):
        return f"({repr(self.s)}, {repr(self.k)}, {repr(self.r)})"


def interaction_hamiltonian_from_k_matrix(k_matrix):
    return interaction_hamiltonian_from_linear_coupling_operator(linear_coupling_operator_from_k_matrix(k_matrix))


def interaction_hamiltonian_from_linear_coupling_operator(l_operator):
    r"""
    Calculate the idealised interaction hamiltonian for the given linear coupling operator.

    .. math::
        H_\text{int}(t) = i(L^T \eta(t)^* - L^\dagger \eta(t)),

    where :math:`\eta = (\eta_1, \eta_1^\dagger; \dots; \eta_m, \eta_m^\dagger)^T`

    TODO: check this properly, not totally sure about the dimensions of :math:`\eta`, not sure if this is correct at all

    Raises `DimensionError` if not ``l_operator`` not a column vector or does not have an even number of rows.
    """
    if l_operator.shape[1] != 1:
        raise DimensionError(f"L is not a column vector: {l_operator.shape}")
    if l_operator.shape[0] % 2 != 0:
        raise DimensionError(f"L does not have even number of rows: {l_operator.shape}")

    from sympy import I, simplify
    states = make_complex_ladder_state(l_operator.shape[1], "eta")
    h_int = I * (l_operator.T * states.conjugate() - l_operator.H * states)
    if h_int.shape != (1, 1):
        raise DimensionError(f"Expected interaction Hamiltonian to be scalar, instead: {h_int.shape}")
    return simplify(h_int[0, 0])


def linear_coupling_operator_from_k_matrix(k_matrix, symbol='a'):
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


def hamiltonian_from_r_matrix(r_matrix, symbol='a'):
    r"""
    Calculate symbolic Hamiltonian from R matrix assuming `paired operator form`.

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


def make_complex_ladder_state(num_dofs, symbol='a'):
    r"""
    Return matrix of complex ladder operators with ``2 * num_dofs`` elements.

    Use ``symbol`` keyword arg to set alternative symbol for variables instead of using ``a``

    For example, for ``num_dofs == 2``, result is :math:`(a_1, a_1^\dagger; a_2, a_2^\dagger)^T`.
    """
    states = []

    for i in range(num_dofs):
        s = Symbol(f"{symbol}_{i + 1}", commutative=False)
        states.append(s)
        states.append(s.conjugate())

    return Matrix(states)


"""Network Synthesis"""


def split_system(open_osc: SLH):
    """
    Split n degree of freedom open oscillator into n one degree of freedom open oscillators and a direct interaction
    Hamiltonian matrix.

    :param open_osc:
    :return: tuple of ([G_1, ..., G_n], H^d)
    """
    dof = open_osc.r.shape[0] // 2

    if dof == 1:  # already separated
        return [open_osc], Matrix.zeros(2, 2)

    # get the list of 2x2 diagonal block matrices of R
    r_blocks = list(map(lambda i: open_osc.r[i:(i+2), i:(i+2)], range(0, 2 * dof, 2)))
    # get the list of mx2 k matrices
    k_blocks = list(map(lambda i: open_osc.k[0:2, i:(i+2)], range(0, 2 * dof, 2)))

    # construct the open oscillators
    gs = list(map(lambda v: SLH(Matrix.eye(2, 2), v[1][0], v[1][1], make_complex_ladder_state(1, f'a_{v[0]}')),
                  enumerate(zip(k_blocks, r_blocks))))

    # TODO: for now we assume no scattering for simplicity

    h_d = Matrix.zeros(dof * 2, dof * 2)

    # construct Hamiltonian interaction matrix
    for j in range(0, dof - 1):
        for k in range(j + 1, dof):
            h_d[(j*2):((j+1)*2), (k*2):((k+1)*2)] = open_osc.r[(k*2):((k+1)*2), (j*2):((j+1)*2)].H \
                - 1 / (2 * I) * (k_blocks[k].H * k_blocks[j] - k_blocks[k].T * k_blocks[j].C)

    return gs, h_d


def split_two_dof(open_osc: SLH):
    r"""
    Split a two dof generalised open oscillator :math:`G` into two one dof generalised open oscillators
    :math:`(G_1, G_2)` which are connected in series and coupled by direct interaction Hamiltonian :math:`H^d_{1,2}`.

    The resulting G_1 and G_2 are assigned Sympy symbols a_1 and a_2 respectively

    :param open_osc: an instance of `SLH` with two degrees of freedom (i.e. r-matrix is 4x4)
    :returns tuple of :math:`(G_1, G_2, H^d_{1,2})`, where :math:`H^d_{1,2}` is given as a matrix
    """

    if open_osc.r.shape != (4, 4) or open_osc.k.shape[1] != 4:
        raise DimensionError("split_two_dof only works on generalised open oscillators with two degrees of freedom")

    # get diagonal 2x2 block matrices of R
    r_1 = open_osc.r[0:2, 0:2]
    r_2 = open_osc.r[2:4, 2:4]

    # get the two mx2 k matrices
    k_1 = open_osc.k[0:2, 0:2]
    k_2 = open_osc.k[0:2, 2:4]

    # TODO: for now we assume no scattering for simplicity

    r_12 = open_osc.r[2:4, 0:2]
    r_21 = open_osc.r[0:2, 2:4]

    h_d = Matrix.zeros(4, 4)

    h_d[0:2, 2:4] = r_21 - 1 / (2*I) * (k_1.H * k_2 - k_1.T * k_2.C)
    h_d[2:4, 0:2] = r_12 - 1 / (2*I) * (k_2.H * k_1 - k_2.T * k_1.C)

    g_1 = SLH(Matrix.eye(2, 2), k_1, r_1, make_complex_ladder_state(1, 'a_1'))
    g_2 = SLH(Matrix.eye(2, 2), k_2, r_2, make_complex_ladder_state(1, 'a_2'))

    return g_1, g_2, h_d
