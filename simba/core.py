from copy import deepcopy
from collections import namedtuple
from sympy import Matrix, BlockDiagMatrix, Symbol, fraction, ImmutableMatrix
from simba.utils import halve_matrix
from simba.errors import DimensionError, CoefficientError, StateSpaceError
from functools import lru_cache


def transfer_function_to_coeffs(expr):
    """
    Extract transfer function coefficients from the given expression which is a function of frequency :math:`s` and a
    ratio of two polynomials.

    Returns a namedtuple containing two lists of length n and n-1 for the numerator and denominator respectively
    in *order of ascending powers*, where n is the order of expr in :math:`s`.

    The numerator list will be padded if the denominator is higher order than the numerator.

    A ``NotImplementedError`` will be raised if the denominator is lower order than the numerator.

    The coefficients are normalised such that the coefficient of the highest order term in the denominator is one.

    :param expr: ratio of two polynomials in :math:`s`
    :return: ``Coefficients`` namedtuple with fields: ``(numer=[b_n, ..., b_0], denom=[a_n, ..., a_1])``
    """
    s = Symbol('s')
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


class StateSpace:
    r"""
    Represents a dynamical state-space which describes the time-domain evolution of a system.

    .. math::
        \dot{x} &= a x + b u, \\
        y &= c x + d u.

    where the state vectors are bounded linear operators usually in `doubled-up form`,

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
         - is_quantum:  whether or not system is quantum.

    Note that the matrices are stored using ``sympy.ImmutableMatrix``, and so are immutable. New `StateSpace`s should
    be created from modified matrices instead.
    """
    def __init__(self, a, b, c, d):
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

        self.is_quantum = False  # used to remember whether system is extended to quantum state-space

    @classmethod
    def from_transfer_function_coeffs(cls, numer, denom):
        r"""
        Return the *quantum* (see [#quantum]_) controllable canonical form state space for the given list of numerators
        and denominators of a pole-zero form transfer function, given in order of ascending powers, assuming complex
        ladder operators :math:`(a, a^\dagger)` are used.

        The coefficients are defined via the transfer function between the input :math:`u(s)` and
        the output :math:`y(s)`, where :math:`s` is the complex Laplace frequency,[#laplace]_

        .. math::
            \frac{y(s)}{u(s)} = \frac{b_0 s^n + b_1 s^{n-1} + \dots + b_{n-1} s + b_n}
            {s^n + a_1 s^{n-1} + \dots + a_{n-1} s + a_n}.

        *Note that we assume the coefficients are normalized with respect to the highest order term in the denominator.*

        Raises `CoefficientError` if lengths of coefficient lists are wrong.

        Reference: https://www.engr.mun.ca/~millan/Eng6825/canonicals.pdf

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
            a[n - 1, i] = +1 * denom_coeff  # would be -1 if used other Laplace convention (s -> -s)

        # construct c matrix
        b_0 = b[-1]
        c = Matrix.zeros(1, n)
        for i, (a_i, b_i) in enumerate(zip(denom, numer)):
            c[0, i] = -1 * (b_i - a_i * b_0)  # would be +1 if other convention

        # construct d matrix
        d = Matrix([b_0])

        return cls(a, b, c, d)

    @classmethod
    def from_transfer_function(cls, expr):
        """Call `from_transfer_function_coeffs` passing the expression to `transfer_function_to_coeffs`."""
        return cls.from_transfer_function_coeffs(*transfer_function_to_coeffs(expr))

    def extended_to_quantum(self):
        """
        Extend to quantum state space to doubled-up ordering (see [#quantum]_).
        Returns extended `StateSpace`. Does not modify original.

        Raises `StateSpaceError` if already quantum.
        """
        if self.is_quantum:
            raise StateSpaceError("System is already quantum.")

        quantum_ss = deepcopy(self)

        quantum_ss.is_quantum = True
        quantum_ss.a = ImmutableMatrix(BlockDiagMatrix(self.a, self.a.C))
        quantum_ss.b = ImmutableMatrix(BlockDiagMatrix(self.b, self.b.C))
        quantum_ss.c = ImmutableMatrix(BlockDiagMatrix(self.c, self.c.C))
        quantum_ss.d = ImmutableMatrix(BlockDiagMatrix(self.d, self.d.C))
        return quantum_ss

    def truncated_to_classical(self):
        """
        Truncate to classical state space from doubled-up ordered state-space (see [#quantum]_).
        Returns truncated `StateSpace`. Does not modify original.

        Raises `StateSpaceError` if not quantum.
        """
        if not self.is_quantum:
            raise StateSpaceError("System is not quantum.")

        classical_ss = deepcopy(self)

        classical_ss.is_quantum = False
        classical_ss.a = halve_matrix(self.a)
        classical_ss.b = halve_matrix(self.b)
        classical_ss.c = halve_matrix(self.c)
        classical_ss.d = halve_matrix(self.d)
        return classical_ss

    def to_transfer_function(self):
        """
        Calculate `SISO` transfer function for the system using the convention given by [#laplace]_.

        Raise `StateSpaceError` if system is not classical.

        Raise `DimensionError` if system has more than one input or output.

        **TODO: work out how to do this for quantum systems**
        """
        if self.is_quantum:
            raise StateSpaceError("Calculating transfer function for quantum systems is not yet implemented.")
        elif self.num_inputs != 1 or self.num_outputs != 1:
            raise DimensionError(f"System is not SISO: num_inputs == {self.num_inputs},"
                                 f"num_outputs == {self.num_outputs}")
        else:
            s = Symbol('s')
            return (self.c * (-s * Matrix.eye(self.a.shape[0]) - self.a).inv() * self.b + self.d)[0, 0]

    @lru_cache()
    def reorder_to_paired_form(self):
        r"""
        Return a new StateSpace with the system matrices reordered so that the state vectors, inputs, and outputs are
        converted from doubled-up form,

        .. math::
            (a_1, a_2, \dots, a_n; a_1^\dagger, a_2^\dagger, \dots, a_n^\dagger)^T,

        to paired operator form,

        .. math::
            (a_1, a_1^\dagger; a_2, a_2^\dagger; \dots; a_n, a_n^\dagger)^T,

        Raise `StateSpaceError` if not quantum.

        Result with be cached using ``functools.lru_cache``, so subsequent calls should be "free".

        :return: StateSpace in paired up form.
        """
        if not self.is_quantum:
            raise StateSpaceError("StateSpace must be quantum.")
        # (1, 2, 3, 11, 22, 33) -> (1, 11, 2, 22, 3, 33)
        n = self.num_degrees_of_freedom
        assert n % 2 == 0, "num_degrees_of_freedom should be even for a quantum system"

        # construct the transformation matrix that reorders the elements
        u = Matrix.zeros(n, n)
        for x in range(n // 2):
            u[x * 2, x] = 1
            u[x * 2 + 1, x + n // 2] = 1

        # apply transformation and return
        return StateSpace(u*self.a*u.inv(), u*self.b*u.inv(), u*self.c*u.inv(), u*self.d*u.inv())

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
        return self.a == other.a and self.b == other.b and self.c == other.c and self.d == other.d and \
            self.is_quantum == other.is_quantum

    def __str__(self):
        """Prettify the equation."""
        from sympy.printing.pretty import pretty
        return f"{pretty(self.a)}\n{pretty(self.b)}\n{pretty(self.c)}\n{pretty(self.d)}\n"

    def __repr__(self):
        return f"{repr(self.a)}\n{repr(self.b)}\n{repr(self.c)}\n{repr(self.d)}\n"

    def __hash__(self):
        return hash(tuple(self))


def transfer_func_coeffs_to_state_space(numer, denom):
    """See `StateSpace.from_transfer_function_coeffs`."""
    return StateSpace.from_transfer_function_coeffs(numer, denom)


def transfer_function_to_state_space(expr):
    """See `StateSpace.from_transfer_function`."""
    return StateSpace.from_transfer_function(expr)
