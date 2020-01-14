
SImBA - Systematic Inference of Bosonic quAntum systems
=======================================================

Welcome to the documentation for simba, a set of python scripts and modules for the systematic synthesis
of linear quantum dynamical systems directly from frequency-domain transfer functions.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    modules/core
    modules/utils
    modules/errors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Glossary
========

.. glossary::

    SISO
        Single-Input Single-Output; systems that only have one input channel and one output channel.

    Doubled-up form
        Where the state-space vectors are ordered such that all the creation operators for the modes follow the
        annihilation operators: :math:`(a_1, \dots, a_n; a_1^\dagger, \dots, a_n^\dagger)^T`, as opposed to a
        paired form: :math:`(a_1, a_1^\dagger; \dots; a_n, a_n^\dagger)^T`.

References
==========

.. |br| raw:: html

    <br />

.. [squeezing-components]
    |br|
    Gough, J. E., James, M. R., & Nurdin, H. I. (2010). Squeezing components in linear quantum feedback networks. Physical Review A - Atomic, Molecular, and Optical Physics, 81(2). https://doi.org/10.1103/PhysRevA.81.023804