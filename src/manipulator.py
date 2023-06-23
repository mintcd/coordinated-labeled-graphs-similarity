from random import shuffle
import networkx as nx

class Manipulator:
  def shuffled_graph(G):
    nodes = list(G.nodes(data=True))
    shuffle(nodes)

    # Create a mapping from original node IDs to shuffled node IDs
    id_mapping = {node_id: shuffled_id for shuffled_id, (node_id, _) in enumerate(nodes)}

    # Create a new graph with shuffled node attributes and modified edges
    shuffled_G = nx.Graph()
    shuffled_G.add_nodes_from((shuffled_id, attr) for shuffled_id, (_, attr) in enumerate(nodes))

    shuffled_edges = [(id_mapping[u], id_mapping[v]) for u, v in G.edges()]
    shuffled_G.add_edges_from(shuffled_edges)

    return shuffled_G