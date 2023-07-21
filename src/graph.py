from random import random, choice, randint, sample, uniform
from itertools import combinations
import numpy as np
from math import sqrt, floor, pi
import networkx as nx
from transform import Transform
import copy
import os
from operations import shuffled_graph, similar_graph


class data:
    def __init__(
        self, original, similar, posision_modified, color_modified, edge_removed
    ):
        self.original = original
        self.similar = similar
        self.posision_modified = posision_modified
        self.color_modified = color_modified
        self.edge_removed = edge_removed


def generate(
    v, c, dim, shuffle=False, same_vectors=False, equidistant=False, write=False
):
    G = nx.Graph()
    G.graph["number_of_colors"] = c
    G.graph["dim"] = dim
    if equidistant:
        v = v - v % (dim + 1)

    combs = list(combinations(list(range(v)), 2))
    edges = sample(combs, randint(1, len(combs)))

    G.add_nodes_from(list(range(v)))
    G.add_edges_from(edges)

    rotation_element = {
        2: np.array([[0, sqrt(3) / 3], [-1 / 2, -sqrt(3) / 6], [1 / 2, -sqrt(3) / 6]]),
        3: np.array(
            [
                [1, 0, -1 / sqrt(2)],
                [-1, 0, -1 / sqrt(2)],
                [0, 1, 1 / sqrt(2)],
                [0, -1, 1 / sqrt(2)],
            ]
        ),
    }

    if equidistant:
        for i in range(floor(v / (dim + 1))):
            params = get_params(dim)
            coors = [Transform.rotate(p, params[0]) for p in rotation_element[dim]]
            for j in range(dim + 1):
                G.nodes[(dim + 1) * i + j]["pos"] = coors[j]
                G.nodes[(dim + 1) * i + j]["color"] = choice(list(range(c)))
    else:
        for _, attr in G.nodes(data=True):
            if same_vectors:
                attr["pos"] = np.array([float(randint(-5, 5)) for _ in range(dim)])
            else:
                attr["pos"] = np.array([float(uniform(-5, 5)) for _ in range(dim)])
            attr["color"] = choice(list(range(c)))

    H = similar_graph(G, get_params(dim))
    if shuffle:
        H = shuffled_graph(H)

    H1 = modify(H, on_position=True)
    H2 = modify(H, on_color=True)
    H3 = modify(H, on_edge=True)

    if write:
        if equidistant:
            if dim == 2:
                folder_name = "dataset\\{}d\\triangles\{}".format(dim, v)
            else:
                folder_name = "dataset\\{}d\\tetrahedra\{}".format(dim, v)
        else:
            folder_name = "dataset\\{}d\\random\{}".format(dim, v)

        os.makedirs(folder_name, exist_ok=True)

        graph_list = [
            ("G.graphml", G),
            ("H.graphml", H),
            ("H1.graphml", H1),
            ("H2.graphml", H2),
            ("H3.graphml", H3),
        ]

        for graph in graph_list:
            for node in graph[1].nodes:
                pos = graph[1].nodes[node]["pos"]
                graph[1].nodes[node]["pos"] = str(pos)
            nx.write_graphml(graph[1], os.path.join(folder_name, graph[0]))

    else:
        return data(G, H, H1, H2, H3)


def load(dim, equidistant, v):
    def revert(G):
        # Revert node IDs back to integers
        node_mapping = {}
        reverted_G = nx.Graph()
        updated_node_attrs = {}

        for old_node in G.nodes:
            new_node = int(old_node)
            node_mapping[old_node] = new_node
            updated_node_attrs[new_node] = G.nodes[old_node].copy()
            updated_node_attrs[new_node]["pos"] = np.fromstring(
                G.nodes[old_node]["pos"][1:-1], sep=" ", dtype=float
            )
            reverted_G.add_node(new_node)

        for old_edge in G.edges:
            new_edge = (node_mapping[old_edge[0]], node_mapping[old_edge[1]])
            reverted_G.add_edge(*new_edge)

        # Set the updated node attributes
        nx.set_node_attributes(reverted_G, updated_node_attrs)
        reverted_G.graph["dim"] = G.graph["dim"]
        reverted_G.graph["number_of_colors"] = G.graph["number_of_colors"]

        return reverted_G

    subfolder = ""
    if equidistant:
        subfolder = "triangles" if dim == 2 else "tetrahedra"
    else:
        subfolder = "random"

    if equidistant:
        v = v - v % (dim + 1)
    folder_name = "dataset\\{}d\\{}\\{}".format(dim, subfolder, v)

    G = revert(nx.read_graphml(os.path.join(folder_name, "G.graphml")))
    H = revert(nx.read_graphml(os.path.join(folder_name, "H.graphml")))
    H1 = revert(nx.read_graphml(os.path.join(folder_name, "H1.graphml")))
    H2 = revert(nx.read_graphml(os.path.join(folder_name, "H2.graphml")))
    H3 = revert(nx.read_graphml(os.path.join(folder_name, "H3.graphml")))

    return data(G, H, H1, H2, H3)


def get_params(dim):
    r = None
    if dim == 2:
        r = uniform(-pi, pi)
        # r = np.array([random(), random()])
        # r = r / np.linalg.norm(r)
    elif dim == 3:
        r = np.array([random(), random(), random(), random()])
        r = r / np.linalg.norm(r)

    else:
        print("Invalid dim (dim in {2,3})")

    k = randint(1, 5)
    if dim == 2:
        t = [randint(1, 5), randint(1, 5)]

    if dim == 3:
        t = [randint(1, 5), randint(1, 5), randint(1, 5)]

    return (r, k, t)


def modify(G: nx.Graph, on_position=False, on_color=False, on_edge=False, rate=0.2):
    H = copy.deepcopy(G)
    if on_position:
        for node in H.nodes:
            if random() < rate:
                H.nodes[node]["pos"] += np.array(
                    list(map(lambda i: choice((-1, 1)), list(range(H.graph["dim"]))))
                )
    if on_color:
        for node in H.nodes:
            if random() < rate:
                H.nodes[node]["color"] = choice(
                    list(range(H.graph["number_of_colors"]))
                )
    if on_edge:
        for edge in H.edges:
            if random() < rate:
                H.remove_edge(edge[0], edge[1])
    return H
