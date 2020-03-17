
from simba import transfer_function_to_graph
from simba.config import temp_set_param, Param
from sympy import Symbol

s = Symbol('s')
tf = (s**3 + s**2 + s - 1) / (-s**3 + s**2 - s - 1)

with temp_set_param('wolframscript', Param.ON):
    transfer_function_to_graph(tf, 'example.pdf', layout='dot')
