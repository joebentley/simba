
from simba import transfer_function_to_graph, transfer_function_to_state_space, split_system
from sympy import Symbol, simplify

s = Symbol('s')

# tf = (s - 1) / (s + 1)
#
# transfer_function_to_graph(tf, 'unstable-filter.pdf')
#
# tf = (s + 1) / (s - 1)
#
# transfer_function_to_graph(tf, 'tuned-cavity.pdf')

tf = (s + 1) / (s - 1)
tf = tf * tf
print(tf)

ss = transfer_function_to_state_space(tf).extended_to_quantum().to_physically_realisable()

print(ss.to_transfer_function()[0, 0])

slh = ss.to_slh()
print(slh)
print()
print(split_system(slh))

transfer_function_to_graph(tf, 'passive-cascade.pdf')