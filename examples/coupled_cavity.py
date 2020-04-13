
from simba import transfer_function_to_graph, tf2rss
from sympy import symbols


# passive realisation (g = 0)
s = symbols('s')
gamma_f, omega_s = symbols('gamma_f omega_s', real=True, positive=True)
tf = (s**2 + s * gamma_f + omega_s**2) / (s**2 - s * gamma_f + omega_s**2)

print(tf2rss(tf).to_slh().split())

print(tf2rss(tf).to_slh().interaction_hamiltonian)

transfer_function_to_graph(tf, 'passive_coupled_cavity.pdf', layout='dot')

# parameterise with lambda = omega_s**2 - g**2 < 0
lmbda = symbols('lambda', real=True, positive=True)
tf = (s**2 + s * gamma_f - lmbda) / (s**2 - s * gamma_f - lmbda)

transfer_function_to_graph(tf, 'active_coupled_cavity.pdf', layout='dot')
