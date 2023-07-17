import generate
from visualize import draw, plot
from infer import similarity_infer
import pprint
import numpy as np
import os
import networkx as nx
import ast

pp = pprint.PrettyPrinter(width=40, compact=True, indent=2)


generate.generate(11, 5, 2, False)

# loaded_G = nx.read_graphml(os.path.join(folder_name, graph[0]))

# # Revert node IDs back to integers
# node_mapping = {}
# reverted_G = nx.Graph()
# for old_node in loaded_G.nodes:
#     new_node = int(old_node)
#     node_mapping[old_node] = new_node
#     reverted_G.add_node(new_node, pos=loaded_G.nodes[old_node]["pos"])

# # Revert edge IDs back to integers
# for old_edge in loaded_G.edges:
#     new_edge = (node_mapping[old_edge[0]], node_mapping[old_edge[1]])
#     reverted_G.add_edge(*new_edge)


# draw(G)
# draw(H)
# draw(h1)
# draw(h2)
# draw(h3)
