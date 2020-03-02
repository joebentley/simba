import simba
import sympy

s = sympy.Symbol('s')
tf = (s**2 + s + 1) / (s**2 - s + 1)

ss = simba.transfer_function_to_state_space(tf)
ss.pprint()
print()
ss = ss.extended_to_quantum().to_physically_realisable()
ss.pprint()

g = ss.to_slh('a')
print(g)
print()

g_1, g_2, h_d = simba.split_two_dof(g)

print(g_1)
print()
print(g_2)
print()
print(h_d)
print()

graph = simba.nodes_from_two_dofs(g_1, g_2, h_d).as_graphviz_agraph()

graph.layout()
graph.draw('example.pdf')
print('wrote example.pdf')
