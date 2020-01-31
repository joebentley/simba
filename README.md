
# SImBA - Systematic Inference of Bosonic quAntum systems

[![](https://img.shields.io/badge/github-joebentley%2Fsimba-brightgreen)](https://github.com/joebentley/simba)
[![](https://img.shields.io/badge/pypi-quantum--simba-brightgreen)](https://pypi.org/project/quantum-simba/)
[![](https://github.com/joebentley/simba/workflows/Python%20application/badge.svg)](https://github.com/joebentley/simba/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/simbapy/badge/?version=latest)](https://simbapy.readthedocs.io/en/latest/?badge=latest)

[>>Documentation<<](https://simbapy.readthedocs.io/en/latest/)

Install via `pip install quantum-simba` (for now)

See `notebooks` for examples.

To clone the dev environment run,

```
$ conda env create -f=environment.yml
$ conda activate simba
```

To install simba for development purposes,

```
$ pwd
... (simba code directory containing setup.py)
$ pip install -e .
(then to run test suite)
$ py.test
```

To build documentation locally,

```
$ pwd
... (simba code directory containing setup.py)
$ ./make_docs.sh
```

## TODO

* Example notebooks
* Write paper
* Upload to PyPI (& Conda?)
