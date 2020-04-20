
# here we provide the public interface

from simba.core import \
    (is_transfer_matrix_physically_realisable,
     transfer_func_coeffs_to_state_space, transfer_function_to_state_space, transfer_function_to_realisable_state_space,
     tf2ss, tf2rss, tf2network, transfer_function_to_coeffs, StateSpace,
     concat, SLH, make_complex_ladder_state, split_system)
from simba.errors import DimensionError, CoefficientError, StateSpaceError
from simba.graph import nodes_from_dofs, transfer_function_to_graph
from simba.utils import adiabatically_eliminate

# Auto-set parameters
import simba.config
config.init_params()
