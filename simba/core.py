from sympy import Matrix


class DimensionError(Exception):
    """
    Represents an error involving matrix dimensions.
    """


class CoefficientError(Exception):
    """
    Represents an error involving transfer function coefficients.
    """


class StateSpace:
    r"""
    Represents a dynamical state-space which describes the time-domain evolution of a system.

    .. math::
        \dot{x} &= a x + b u, \\
        y &= c x + d u.

    where the state vectors are,

    .. math::
        x \in \mathbb{C}^{n\times 1},\
        u \in \mathbb{C}^{m\times 1},\
        y \in \mathbb{C}^{l\times 1},

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
    """
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

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
                                 f"a - {na}, b - {nb}, c - {nc}")
        if not lc == ld:
            raise DimensionError(f"Number of output channels for matrix c not equal to matrix d: {lc} != {ld}")

    @classmethod
    def from_transfer_function_coeffs(cls, numer, denom):
        r"""
        Return the *quantum* (see [#quantum]_) controllable canonical form state space for the given list of numerators
        and denominators of a pole-zero form transfer function, assuming complex ladder operators :math:`(a, a^\dagger)`
        are used.

        The coefficients are defined via the transfer function between the input :math:`u(s)` and
        the output:math:`y(s)`, where :math:`s` is the complex Laplace frequency,

        .. math::
            \frac{y(s)}{u(s)} = \frac{b_0 s^n + b_1 s^{n-1} + \dots + b_{n-1} s + b_n}
            {s^n + a_1 s^{n-1} + \dots + a_{n-1} s + a_n}.

        *Note that we assume the coefficients are normalized with respect to the highest order term in the denominator.*

        Raises `CoefficientError` if lengths of coefficient lists are wrong.

        Reference: https://www.engr.mun.ca/~millan/Eng6825/canonicals.pdf

        :param numer: The numerator coefficients: :math:`[b_0, \dots, b_n]`
        :param denom: The denominator coefficients: :math:`[a_1, \dots, a_n]`

        :return: StateSpace for the given system
        """

        pass


def transfer_func_coeffs_to_state_space(numer, denom):
    """
    See `simba.core.StateSpace.from_transfer_function_coeffs`
    """
    return StateSpace.from_transfer_function_coeffs(numer, denom)
