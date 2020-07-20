
from enum import Enum
from typing import List, Set
from .core import SplitNetwork


class ConnectionType(Enum):
    r"""
    Represents a connection between two main modes in the setup.

    It can either be beamsplitter-like (i.e. :math:`\hat{a}\hat{b}^\dagger + \hat{a}^\dagger\hat{b}`)
    or non-energy conserving squeezing-style interaction (i.e. :math:`\hat{a}\hat{b} + \hat{a}^\dagger\hat{b}^\dagger`).
    """
    BS = 1        # beamsplitter-like energy-preserving connection
    SQZ = 2       # squeezing style interaction, either two-mode (non-degenerate) or single-mode (degenerate)

    def __str__(self):
        if self == ConnectionType.BS:
            return 'BS'
        elif self == ConnectionType.SQZ:
            return 'SQZ'


class Connection:
    """
    Represents a single connection to the node at given index (starting at zero).
    """
    def __init__(self, index: int, connection_type: ConnectionType):
        self.index = index
        self.connection_type = connection_type


class Internal(Enum):
    """Represents the type of internal dynamics of the system."""
    TUNED = 1     # tuned cavity
    DETUNED = 2   # detuned cavity
    DPA = 3       # degenerate parametric amplifier (internal squeezing)
    ALL = 4       # both detuned and internal squeezing


class Node:
    """
    Represents a single generalised open oscillator.

    ``self.connections`` is a list of `Connection` to other generalised open oscillators, not including the series connection
    ``self.self_connections`` is a set of `ConnectionType` for connecting this node to itself, for realising the K matrix
    """
    def __init__(self, internal=Internal.TUNED):
        self.internal = internal
        self.connections = []
        self.self_connections = set()

    def get_connections_to(self, index: int) -> set:
        """Filter for *set* of `ConnectionType` to the `Node` with given index."""
        return set(map(lambda conn: conn.connection_type, filter(lambda conn: conn.index == index, self.connections)))

    @property
    def is_series_connected(self) -> bool:
        """
        If the auxiliary mode is not coupled to the main mode, then the main mode is not connected to the series
        connections, so we can ignore it in the series connections.
        """
        return len(self.self_connections) != 0

    def __str__(self) -> str:
        self_conns_str = "-> self via "
        if ConnectionType.BS in self.self_connections:
            self_conns_str += "BS"
        if ConnectionType.SQZ in self.self_connections:
            self_conns_str += "and SQZ"

        connections_str = ""

        for connection in self.connections:
            connections_str += f"-> {connection.index} via {connection.connection_type}\n"

        return self_conns_str + '\n' + connections_str


def _arrow_head_from_connections_set(connections: Set[ConnectionType]) -> str:
    """Determine arrowhead from set of `ConnectionType`."""
    arrow_head = None
    if ConnectionType.SQZ in connections and ConnectionType.BS in connections:
        arrow_head = 'lodiamondrbox'
    elif ConnectionType.BS in connections:
        arrow_head = 'odiamond'
    elif ConnectionType.SQZ in connections:
        arrow_head = 'obox'
    return arrow_head


class Nodes:
    """
    List of Nodes connected in series, indexes starting at zero.
    """
    def __init__(self, nodes: List[Node] = None):
        self.nodes = nodes or []

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        s = ""
        for i, node in enumerate(self.nodes):
            s += f"node {i + 1}\n{str(node)}"
        return s

    def as_graphviz_agraph(self):
        """Convert to a pygraphviz.AGraph object for display."""
        import pygraphviz as pgv

        g = pgv.AGraph(strict=False, directed=True)

        for i, node in enumerate(self.nodes):
            node_shape = 'circle'
            if node.internal == Internal.TUNED:
                node_shape = 'circle'
            elif node.internal == Internal.DETUNED:
                node_shape = 'ellipse'
            elif node.internal == Internal.DPA:
                node_shape = 'square'
            elif node.internal == Internal.ALL:
                node_shape = 'triangle'

            g.add_node(str(i + 1), shape=node_shape)

            # determine self connection shapes
            if len(node.self_connections) != 0:
                g.add_node(f"{i + 1}'", shape='point', width=0.1)
                # g.add_node(str(i + 1), shape=node_shape)  # reset main node shape

                arrow_head = _arrow_head_from_connections_set(node.self_connections)

                if arrow_head is not None:
                    g.add_edge(str(i + 1), f"{i + 1}'", arrowtail=arrow_head, arrowhead=arrow_head, dir='both')

        # add series connection between each node if series connected
        first_connected = None
        last_connected = None
        connection_list = []

        for i, node in enumerate(self.nodes):
            if node.is_series_connected:
                if first_connected is None:
                    first_connected = i
                elif last_connected is None:
                    last_connected = i

                connection_list.append(i)

        if last_connected is None:
            last_connected = first_connected

        for i in range(0, len(connection_list) - 1):
            g.add_edge(str(connection_list[i] + 1) + "'", str(connection_list[i + 1] + 1) + "'",
                       'series', arrowtail='none', arrowhead='normal', dir='both')

        # add other connections between each node
        for i, node in enumerate(self.nodes):
            for other in range(i+1, len(self.nodes)):
                connections = node.get_connections_to(other)

                arrow_head = _arrow_head_from_connections_set(connections)

                if arrow_head is not None:
                    # add edge depending on whether or not edge already exists in that direction
                    if not g.has_edge(str(i + 1), str(other + 1)):
                        g.add_edge(str(i + 1), str(other + 1), 'interaction', arrowtail=arrow_head, arrowhead=arrow_head, dir='both')
                    else:
                        g.add_edge(str(other + 1), str(i + 1), 'interaction', arrowtail=arrow_head, arrowhead=arrow_head, dir='both')

        # add input and output nodes
        g.add_node('input', shape='plaintext')
        g.add_node('output', shape='plaintext')

        g.add_edge('input', str(first_connected + 1) + "'")
        g.add_edge(str(last_connected + 1) + "'", 'output')

        return g


def nodes_from_dofs(gs, h_d) -> Nodes:
    """
    Construct the Node graph for an n degree-of-freedom generalised open oscillator
    :param gs: list of n 1-dof generalised open oscillators
    :param h_d: the direct interaction Hamiltonian matrix
    :return: a `Nodes` instance
    """
    from sympy import re, simplify

    def check_dpa_or_tuned(g):
        detuning = False
        internal_squeezing = False
        if simplify(g.r[0, 0]) != 0:
            detuning = True
        if simplify(g.r[0, 1]) != 0:
            internal_squeezing = True
        if not detuning and not internal_squeezing:
            return Internal.TUNED
        if detuning and not internal_squeezing:
            return Internal.DETUNED
        if not detuning and internal_squeezing:
            return Internal.DPA
        if detuning and internal_squeezing:
            return Internal.ALL

    nodes = list(map(Node, map(check_dpa_or_tuned, gs)))

    # now we figure out which self-connection we need
    # see Hendra (2008) Section 6.3 https://arxiv.org/abs/0806.4448
    for i, g in enumerate(gs):
        self_conn = g.k
        has_beamsplitter_mixing = self_conn[0] != 0
        has_squeezing_process = self_conn[1] != 0

        if has_beamsplitter_mixing:
            nodes[i].self_connections.add(ConnectionType.BS)

        if has_squeezing_process:
            nodes[i].self_connections.add(ConnectionType.SQZ)

    # figure out the connections to other matrices from the interaction Hamiltonian matrix
    # it should be Hermitian and off-diagonal so we'll just look at the upper-right block
    dof = len(gs)
    for j in range(0, dof - 1):
        for k in range(j + 1, dof):
            h_d_part = h_d[(j*2):((j+1)*2), (k*2):((k+1)*2)]

            # again can just look at one column as the interaction is symmetric
            h_d_part = h_d_part[:, 0].T

            # TODO: check this more thoroughly
            if h_d_part[0] != 0:
                nodes[j].connections.append(Connection(k, ConnectionType.BS))
                nodes[k].connections.append(Connection(j, ConnectionType.BS))
            if h_d_part[1] != 0:
                nodes[k].connections.append(Connection(j, ConnectionType.SQZ))
                nodes[j].connections.append(Connection(k, ConnectionType.SQZ))

    return Nodes(nodes)


def nodes_from_network(network: SplitNetwork) -> Nodes:
    """Call `nodes_from_dofs` with contents of network."""
    return nodes_from_dofs(network.gs, network.h_d)


def transfer_function_to_graph(tf, filename, *, layout='neato'):
    """
    Directly convert SISO transfer function to graph.

    Examples:
        >>>from sympy import symbols
        >>>s = symbols('s')
        >>>gamma = symbols('gamma', real=True, positive=True)
        >>>tf = (s - gamma) / (s + gamma)
        >>>transfer_function_to_graph(tf, 'unstable-filter.pdf')
    """
    from simba import transfer_function_to_state_space, split_system

    ss = transfer_function_to_state_space(tf).extended_to_quantum().to_physically_realisable()

    g = nodes_from_dofs(*split_system(ss.to_slh())).as_graphviz_agraph()
    g.layout(prog=layout)
    g.draw(filename)
    print(f"wrote {filename}")
