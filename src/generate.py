from random import random, choice, randint, sample
from itertools import combinations
import numpy as np
from math import sqrt, floor
import networkx as nx
from transform import Transform2D
from transform import similar_graph
from copy import deepcopy
import os


def generate(v, c, dim, equidistant=False):
    G = nx.Graph()
    G.v = v
    G.c = c
    if equidistant:
        v = v - v % 3
    combs = list(combinations(list(range(v)), 2))
    edges = sample(combs, randint(1, len(combs) - 1))

    if equidistant:
        G.add_nodes_from(list(range(v)))
        equil = np.array(
            [[0, sqrt(3) / 3], [-1 / 2, -sqrt(3) / 6], [1 / 2, -sqrt(3) / 6]]
        )
        for i in range(floor(v / 3)):
            params = get_params(dim)
            coors = [Transform2D.rotate(p, params[0]) for p in equil]
            G.nodes[3 * i]["pos"] = coors[0]
            G.nodes[3 * i]["color"] = choice(list(range(c)))
            G.nodes[3 * i + 1]["pos"] = coors[1]
            G.nodes[3 * i + 1]["color"] = choice(list(range(c)))
            G.nodes[3 * i + 2]["pos"] = coors[2]
            G.nodes[3 * i + 2]["color"] = choice(list(range(c)))

        G.add_edges_from(combs)
    else:
        # Generate random coordinates
        G.add_nodes_from(list(range(v)))
        G.add_edges_from(edges)
        for _, attr in G.nodes(data=True):
            attr["pos"] = [float(randint(1, 10)) for _ in range(dim)]
            attr["color"] = choice(list(range(c)))

    H = similar_graph(G, get_params(dim))

    h1 = modify_positions(H)
    h2 = modify_colors(H)
    h3 = remove_edges(H)

    if equidistant:
        folder_name = "dataset\\2d\\triangles\{}".format(v)
    else:
        folder_name = "dataset\\2d\\random\{}".format(v)

    os.makedirs(folder_name, exist_ok=True)

    graph_list = [
        ("G.graphml", G),
        ("H.graphml", H),
        ("H1.graphml", h1),
        ("H2.graphml", h2),
        ("H3.graphml", h3),
    ]

    for graph in graph_list:
        for node in graph[1].nodes:
            pos = graph[1].nodes[node]["pos"]
            graph[1].nodes[node]["pos"] = str(pos)
        nx.write_graphml(graph[1], os.path.join(folder_name, graph[0]))


def get_params(dim):
    r = None
    if dim == 2:
        r = np.array([random(), random()])
        r = r / np.linalg.norm(r)
        # print("Rotation by complex", r[0], "+", r[1] if r[1] >= 0 else r[1], "i")
    elif dim == 3:
        a = random()
        b = random() * (1 - a * a)
        c = random() * (1 - a * a - b * b)
        d = 1 - a * a - b * b - c * c
        axis = np.array(
            [a, choice([-1, 1]) * b + choice([-1, 1]) * c - choice([-1, 1]) * d]
        )
        img = (b, c, d)
        # print("Rotation by quaternion", a, "+", img)
    else:
        print("Invalid dim (dim in {2,3})")

    k = randint(1, 10)
    # print("Scale by", k)

    if dim == 2:
        t = [randint(1, 10), randint(1, 10)]

    if dim == 3:
        t = [randint(1, 10), randint(1, 10), randint(1, 10)]
    # print("Translate by", t)
    return (r, k, t)


def modify_positions(H, rate=0.2):
    h = deepcopy(H)
    for node in h.nodes:
        if random() < rate:
            h.nodes[node]["pos"] += np.array([choice((-1, 1)), choice((-1, 1))])
    return h


def modify_colors(H, rate=0.2):
    h = deepcopy(H)
    for node in h.nodes:
        if random() < rate:
            h.nodes[node]["color"] = choice(list(range(h.c)))
    return h


def remove_edges(H: nx.Graph, rate=0.2):
    h = deepcopy(H)
    for edge in h.edges:
        if random() < rate:
            h.remove_edge(edge[0], edge[1])
    return h
