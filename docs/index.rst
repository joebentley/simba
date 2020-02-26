
SImBA - Systematic Inference of Bosonic quAntum systems
=======================================================

|github link| |CI status| |License: MIT| |Documentation Status|

.. |github link| image:: https://img.shields.io/badge/github-joebentley%2Fsimba-brightgreen
   :target: https://github.com/joebentley/simba
.. |pypi| image:: https://img.shields.io/badge/pypi-quantum--simba-brightgreen
   :target: https://pypi.org/project/quantum-simba/
.. |CI status| image:: https://github.com/joebentley/simba/workflows/Python%20application/badge.svg
   :target: https://github.com/joebentley/simba/actions
.. |License: MIT| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
.. |Documentation Status| image:: https://readthedocs.org/projects/simbapy/badge/?version=latest
   :target: https://simbapy.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


Welcome to the documentation for simba, a set of python scripts and modules for the systematic synthesis
of linear quantum dynamical systems directly from frequency-domain transfer functions.

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    modules/core
    modules/utils
    modules/errors
    modules/graph

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
        `paired operator form`: :math:`(a_1, a_1^\dagger; \dots; a_n, a_n^\dagger)^T`.

    Paired operator form
        Where each pair of operators corresponds to the same mode, in the form
        :math:`(a_1, a_1^\dagger; \dots; a_n, a_n^\dagger)^T`, in contrast to `doubled-up form`.

References
==========

.. |br| raw:: html

    <br />

.. [squeezing-components]
    |br|
    Gough, J. E., James, M. R., & Nurdin, H. I. (2010). Squeezing components in linear quantum feedback networks. Physical Review A - Atomic, Molecular, and Optical Physics, 81(2). https://doi.org/10.1103/PhysRevA.81.023804

.. [synthesis]
    |br|
    Nurdin, Hendra I., Matthew R. James, and Andrew C. Doherty. "Network Synthesis of Linear Dynamical Quantum Stochastic Systems." SIAM Journal on Control and Optimization 48.4 (2009): 2686–2718. Crossref. Web. https://arxiv.org/abs/0806.4448

.. [transfer-function]
    |br|
    A. J. Shaiju and I. R. Petersen, "A Frequency Domain Condition for the Physical Realizability of Linear Quantum Systems," in IEEE Transactions on Automatic Control, vol. 57, no. 8, pp. 2033-2044, Aug. 2012. https://doi.org/10.1109/TAC.2012.2195929