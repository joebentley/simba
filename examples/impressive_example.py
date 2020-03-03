
from simba import transfer_function_to_graph
from sympy import Symbol

s = Symbol('s')
tf = (s**2 + s + 1) / (s**2 - s + 1)

transfer_function_to_graph(tf, 'example.pdf')
