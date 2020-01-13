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
