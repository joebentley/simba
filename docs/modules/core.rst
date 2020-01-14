==========================
``core.py``: Core features
==========================

.. automodule:: simba.core
    :members:

.. rubric:: Footnotes

.. [#quantum]
    Quantum in this case meaning that the following transformation is applied:

    .. math::
        \dot{x} = a x + b u,\quad y = c x + d u

    to,

    .. math::
        \begin{bmatrix}\dot{x} \\ \dot{x}^\#\end{bmatrix} &= \begin{bmatrix}a & 0 \\ 0 & a^\#\end{bmatrix}
        \begin{bmatrix}x \\ x^\#\end{bmatrix}
        + \begin{bmatrix}b & 0 \\ 0 & b^\#\end{bmatrix} \begin{bmatrix}u \\ u^\#\end{bmatrix}, \\
        \begin{bmatrix}y \\ y^\#\end{bmatrix} &= \begin{bmatrix}c & 0 \\ 0 & c^\#\end{bmatrix}
        \begin{bmatrix}x \\ x^\#\end{bmatrix}
        + \begin{bmatrix}d & 0 \\ 0 & d^\#\end{bmatrix} \begin{bmatrix}u \\ u^\#\end{bmatrix},

    where for a matrix :math:`m`, the notation :math:`m^\#` means "take the adjoint of each element". Effectively
    each vector is in `doubled-up form`, as discussed in [squeezing-components]_.