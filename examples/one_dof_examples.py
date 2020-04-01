from simba import transfer_function_to_graph, transfer_function_to_state_space
from sympy import symbols

s = symbols('s')
gamma = symbols('gamma', real=True, positive=True)

tf = (s - gamma) / (s + gamma)

transfer_function_to_graph(tf, 'unstable-filter.pdf')

tf = (s + gamma) / (s - gamma)

transfer_function_to_graph(tf, 'tuned-cavity.pdf')

tf = (s + gamma) / (s - gamma)
tf = tf * tf

# FIXME: should just be a pure cascade realisation

transfer_function_to_graph(tf, 'passive-cascade.pdf')