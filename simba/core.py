from sympy import Matrix


class DimensionError(Exception):
    """
    Represents an error involving matrix dimensions
    """
    pass


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

        # check dimensions
        if not a.is_square:
            raise DimensionError("a matrix is not square")
        n = a.shape[0]
        # m =
        # if not b.shape[0] == num_dof

    @classmethod
    def from_transfer_function_coeffs(cls, numer, denom):
        """
        Return the controllable canonical form state space for the given list of numerators and denominators.

        Reference: https://www.engr.mun.ca/~millan/Eng6825/canonicals.pdf

        :param numer: list of
        :param denom: hello

        :return: the thing
        """
        pass


def transfer_func_coeffs_to_state_space(numer, denom):
    """
    See `qsynth.core.StateSpace.from_transfer_function_coeffs`
    """
    return StateSpace.from_transfer_function_coeffs(numer, denom)
