
import enum
from contextlib import contextmanager


class Param(enum.Enum):
    """Enum representing a parameter. Param.ON is truthy, everything else is falsey."""
    ON = 0    # force on
    OFF = 1   # force off
    AUTO = 2  # set using init_params

    def __bool__(self):
        return self == Param.ON


params = {
    # run slow but important checks to verify results
    'checks': Param.ON,
    # use wolframscript for more intensive symbolic manipulations
    'wolframscript': Param.OFF
}


def is_wolframscript_installed():
    """Check whether wolframscript is installed and in PATH."""
    from shutil import which
    return which("wolframscript") is not None


def init_params():
    """Check what to set params to if they are set to "auto"."""
    if params['wolframscript'] == Param.AUTO and is_wolframscript_installed():
        params['wolframscript'] = Param.ON

    if params['wolframscript'] == Param.ON and params['checks'] == Param.AUTO:
        params['checks'] = Param.ON


@contextmanager
def temp_set_param(param, to):
    """
    Use via ``with`` to temporarily set param to given value, changing it back afterwards.

    E.g.

    .. code-block:: python

        with temp_set_param('wolframscript', Param.ON):
            # do stuff using wolframscript
        # wolframscript now disabled
    """
    previous = params[param]
    params[param] = to
    yield
    params[param] = previous


@contextmanager
def temp_set_params(params_to_merge):
    """
    Same as `temp_set_param` but params_to_merge is a dictionary of params to `Param` values to update
    """
    from copy import deepcopy
    previous = deepcopy(params)
    params.update(params_to_merge)
    yield
    params.update(previous)
