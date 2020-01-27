
# here we provide the public interface

from simba.core import \
    (is_transfer_matrix_physically_realisable,
     transfer_func_coeffs_to_state_space, transfer_function_to_state_space,
     transfer_function_to_coeffs, StateSpace,
     concat, SLH)
from simba.errors import DimensionError, CoefficientError, StateSpaceError
