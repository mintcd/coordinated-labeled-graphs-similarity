import networkx as nx
import random
from transform import Transform
import copy


def shuffled_graph(G: nx.Graph):
    n = G.number_of_nodes()
    permutation = list(range(0, n))
    for i in range(n - 1, 0, -1):
        j = random.randint(0, i)
        permutation[i], permutation[j] = permutation[j], permutation[i]

    H = nx.Graph()
    H.graph["dim"] = G.graph["dim"]
    H.graph["number_of_colors"] = G.graph["number_of_colors"]

    H.add_nodes_from(list(range(n)))
    for i in range(n):
        H.nodes[permutation[i]]["pos"] = G.nodes[i]["pos"]
        H.nodes[permutation[i]]["color"] = G.nodes[i]["color"]
    for edge in G.edges():
        H.add_edge(permutation[edge[0]], permutation[edge[1]])

    return H


def similar_graph(G, params):
    H = copy.deepcopy(G)
    for _, attr in H.nodes(data=True):
        attr["pos"] = Transform.similar(attr["pos"], params)
    return H
