
from simba import transfer_function_to_graph, transfer_function_to_state_space
from sympy import Symbol

s = Symbol('s')

tf = (s - 1) / (s + 1)

transfer_function_to_graph(tf, 'unstable-filter.pdf')

tf = (s + 1) / (s - 1)

transfer_function_to_graph(tf, 'tuned-cavity.pdf')

tf = (s - 1) / (s + 1)
tf = tf * tf


transfer_function_to_graph(tf, 'passive-cascade.pdf')