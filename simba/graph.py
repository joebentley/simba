
from sympy import Matrix
from enum import Enum


class ConnectionType(Enum):
    BS = 1        # beamsplitter-like energy-preserving connection
    SQZ = 2       # squeezing style interaction, either two-mode (non-degenerate) or single-mode (degenerate)

    def __str__(self):
        if self == ConnectionType.BS:
            return 'BS'
        elif self == ConnectionType.SQZ:
            return 'SQZ'


class Connection:
    """
    Represents a single connection to the node at given index (starting at zero)
    """
    def __init__(self, index: int, connection_type: ConnectionType):
        self.index = index
        self.connection_type = connection_type


class Internal(Enum):
    TUNED = 1     # tuned cavity
    DETUNED = 2   # detuned cavity
    DPA = 3       # degenerate parametric amplifier (internal squeezing)
    ALL = 4       # both detuned and internal squeezing


class Node:
    """
    Represents a single generalised open oscillator.

    ``self.connections`` is a list of `Connection` to other generalised open oscillators
    ``self.self_connections`` is a set of `ConnectionType` for connecting this node to itself, for realising the K matrix
    """
    def __init__(self, internal=Internal.TUNED):
        self.internal = internal
        self.connections = []
        self.self_connections = set()

    def get_connections_to(self, index) -> set:
        """Filter for *set* of `ConnectionType` to the `Node` with given index."""
        return set(map(lambda conn: conn.connection_type, filter(lambda conn: conn.index == index, self.connections)))

    def __str__(self):
        self_conns_str = "-> self via "
        if ConnectionType.BS in self.self_connections:
            self_conns_str += "BS"
        if ConnectionType.SQZ in self.self_connections:
            self_conns_str += "and SQZ"

        connections_str = ""

        for connection in self.connections:
            connections_str += f"-> {connection.index} via {connection.connection_type}\n"

        return self_conns_str + '\n' + connections_str


def _arrow_head_from_connections_set(connections):
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
    def __init__(self, nodes=None):
        self.nodes = nodes or []

    def __iter__(self):
        return iter(self.nodes)

    def __str__(self):
        s = ""
        for i, node in enumerate(self.nodes):
            s += f"node {i}\n{str(node)}"
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
                node_shape = 'diamond'
            elif node.internal == Internal.ALL:
                node_shape = 'star'

            g.add_node(str(i), shape=node_shape)

            # determine self connection shapes
            if len(node.self_connections) != 0:
                g.add_node(f"{i}'", shape=node_shape)
                g.add_node(str(i), shape='circle')  # reset main node shape

                arrow_head = _arrow_head_from_connections_set(node.self_connections)

                if arrow_head is not None:
                    g.add_edge(str(i), f"{i}'", arrowtail=arrow_head, arrowhead=arrow_head, dir='both')

        # add other connections between each node
        for i, node in enumerate(self.nodes):
            for other in range(i+1, len(self.nodes)):
                connections = node.get_connections_to(other)

                arrow_head = _arrow_head_from_connections_set(connections)

                if arrow_head is not None:
                    g.add_edge(str(i), str(other), 'interaction', arrowtail=arrow_head, arrowhead=arrow_head, dir='both')
                    g.add_edge(str(other), str(i), 'interaction', arrowtail=arrow_head, arrowhead=arrow_head, dir='both')

        # add series connection between each node
        for i in range(0, len(self.nodes) - 1):
            g.add_edge(str(i), str(i + 1), 'series', arrowtail='none', arrowhead='normal', dir='both')

        # add input and output nodes
        g.add_node('input', shape='plaintext')
        g.add_node('output', shape='plaintext')

        g.add_edge('input', '0')
        g.add_edge(str(len(self.nodes) - 1), 'output')

        return g


def nodes_from_two_dofs(g_1, g_2, h_d):
    """
    Construct the Node graph for a two degree-of-freedom generalised open oscillator
    :param g_1: the first generalised open oscillator feeding its output into g_2
    :param g_2: the second generalised open oscillator taking its input from g_1
    :param h_d: the direct interaction Hamiltonian between g_1 and g_2
    :return: a `Nodes` instance
    """

    # TODO: distinguish between detuned cavity and tuned DPA
    node_1 = Node(Internal.ALL if g_1.r != Matrix.zeros(2, 2) else Internal.TUNED)
    node_2 = Node(Internal.ALL if g_2.r != Matrix.zeros(2, 2) else Internal.TUNED)

    # now we figure out which self-connection we need
    # only need to look at first column, as discussed by Hendra (2008) Section 6.3 https://arxiv.org/abs/0806.4448
    self_conn_1 = g_1.k[:, 0]

    if self_conn_1[0] != 0:
        node_1.self_connections.add(ConnectionType.BS)
    if self_conn_1[1] != 0:
        node_1.self_connections.add(ConnectionType.SQZ)

    self_conn_2 = g_1.k[:, 0]

    if self_conn_2[0] != 0:
        node_2.self_connections.add(ConnectionType.BS)
    if self_conn_1[1] != 0:
        node_2.self_connections.add(ConnectionType.SQZ)

    # figure out the connections to other matrices from the interaction Hamiltonian matrix
    # it should be Hermitian and off-diagonal so we'll just look at the upper-right block
    h_d = h_d[0:2, 2:4]

    # again can just look at one column as the interaction is symmetric
    h_d = h_d[:, 0].T

    # TODO: check this more thoroughly
    if h_d[0] != 0:
        node_1.connections.append(Connection(1, ConnectionType.BS))
        node_2.connections.append(Connection(0, ConnectionType.BS))
    if h_d[1] != 0:
        node_1.connections.append(Connection(1, ConnectionType.SQZ))
        node_2.connections.append(Connection(0, ConnectionType.SQZ))

    return Nodes([node_1, node_2])


def two_dof_transfer_function_to_graph(tf, filename):
    """Directly convert two dof transfer function to graph."""
    from simba import transfer_function_to_state_space, split_two_dof

    ss = transfer_function_to_state_space(tf).extended_to_quantum().to_physically_realisable()

    g = nodes_from_two_dofs(*split_two_dof(ss.to_slh('a'))).as_graphviz_agraph()
    g.layout()
    g.draw(filename)
    print(f"wrote {filename}")
