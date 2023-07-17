import copy
import numpy as np
import networkx as nx


def similar_graph(G, params):
    H = copy.deepcopy(G)
    for _, attr in H.nodes(data=True):
        attr["pos"] = Transform2D.similar(attr["pos"], params)

    return H


class Transform2D:
    def translate(vec, by):
        return vec + by

    def rotate(vec, by):
        rotated = complex(vec[0], vec[1]) * complex(by[0], by[1])
        return np.array([rotated.real, rotated.imag])

    def scale(vec, k):
        return vec * k

    def similar(vec, params):
        return Transform2D.rotate(vec, params[0]) * params[1] + params[2]
