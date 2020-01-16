
# here we provide the public interface

from simba.core import \
    (transfer_function_to_coeffs, StateSpace,
     transfer_func_coeffs_to_state_space, transfer_function_to_state_space)
from simba.errors import DimensionError, CoefficientError, StateSpaceError
