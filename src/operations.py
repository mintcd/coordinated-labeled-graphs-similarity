import networkx as nx
import random
from transform import Transform
import copy
import numpy as np


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


def normalized_clouds(g_cloud, h_cloud):
    g_centroid = sum([g_cloud[id]["pos"] for id in g_cloud]) / len(g_cloud)
    for id in g_cloud:
        g_cloud[id]["pos"] -= g_centroid
    h_centroid = sum([h_cloud[id]["pos"] for id in h_cloud]) / len(h_cloud)
    for id in h_cloud:
        h_cloud[id]["pos"] -= h_centroid

    g_norms = sorted([np.linalg.norm(g_cloud[id]["pos"]) for id in g_cloud])
    h_norms = sorted([np.linalg.norm(h_cloud[id]["pos"]) for id in h_cloud])

    if len(g_norms) != len(h_norms):
        print("Different number of norms")
        return False

    ratios = list(map(lambda x, y: x / y, g_norms, h_norms))

    if not all(np.isclose(ratio, ratios[0]) for ratio in ratios):
        print("Invalid: Lists of norms have different ratios")
        return None, None

    for id in h_cloud:
        h_cloud[id]["pos"] *= ratios[0]

    return g_cloud, h_cloud
