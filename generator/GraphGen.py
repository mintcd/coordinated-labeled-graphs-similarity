import networkx as nx
from random import randint, random, choice, sample, randrange, uniform, shuffle
import numpy as np
from scipy.spatial.transform import Rotation
import json
from math import pi, sin, cos
from itertools import combinations
from copy import deepcopy
from pymongo import MongoClient

SCALE_LIMIT = 10
TRANS_LIMIT = 100
POINT_LIMIT = 100
NODE_LIMIT = 100
DIMENSION = 3


class ParamGen:
    def random():
        params = {}

        params["rotation"] = {}
        params["rotation"]["angles"] = list(
            map(lambda i: uniform(-pi, pi), [0] * DIMENSION)
        )
        params["rotation"]["order"] = sample([0, 1, 2], 3)

        params["scaling"] = uniform(-SCALE_LIMIT, SCALE_LIMIT)

        params["translation"] = (
            uniform(-TRANS_LIMIT, TRANS_LIMIT),
            uniform(-TRANS_LIMIT, TRANS_LIMIT),
            uniform(-TRANS_LIMIT, TRANS_LIMIT),
        )

        return params


class MatrixGen:
    def __rotationX(params):
        alpha = params["rotation"]["angles"][0]

        return np.array(
            [[cos(alpha), -sin(alpha), 0], [sin(alpha), cos(alpha), 0], [0, 0, 1]]
        )

    def __rotationY(params):
        beta = params["rotation"]["angles"][1]
        return np.array(
            [[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]]
        )

    def __rotationZ(params):
        gamma = params["rotation"]["angles"][2]
        return np.array(
            [[1, 0, 1], [0, cos(gamma), -sin(gamma)], [0, sin(gamma), cos(gamma)]]
        )

    def rotation(params):
        order = params["rotation"]["order"]
        matrices = []

        def appendMatrix(matrices, number):
            if number == 0:
                matrices.append(MatrixGen.__rotationX(params))
            if number == 1:
                matrices.append(MatrixGen.__rotationY(params))
            if number == 2:
                matrices.append(MatrixGen.__rotationZ(params))

        [appendMatrix(matrices, number) for number in order]

        rotationMatrix = np.eye(4)
        rotationMatrix[:3, :3] = np.matmul(
            matrices[2], (np.matmul(matrices[1], matrices[0]))
        )
        rotationMatrix[3, 3] = 1

        return rotationMatrix

    def similarity(params):
        # Assign the rotation matrix
        similarityMatrix = MatrixGen.rotation(params)

        # Add scaling factor
        similarityMatrix[:3, :3] *= params["scaling"]

        # Add translation
        translationVector = np.array(params["translation"])
        similarityMatrix[:3, 3] = translationVector

        return similarityMatrix


class Graph:
    def gen(nodeNum=None, edgeNum=None):
        graph = nx.Graph()
        if not nodeNum:
            nodeNum = randint(2, NODE_LIMIT)
        if not edgeNum:
            edgeNum = randint(0, int(nodeNum / 2 * (nodeNum - 1)))

        coordinates = [
            [randint(0, 100) for _ in range(DIMENSION)] + [1] for _ in range(nodeNum)
        ]
        graph.add_nodes_from([(i, {"pos": coordinates[i]}) for i in range(nodeNum)])

        possible_edges = list(combinations(range(nodeNum), 2))
        shuffle(possible_edges)
        graph.add_edges_from(possible_edges[:edgeNum])

        return graph

    def similarize(graph: nx.Graph, params):
        matrix = MatrixGen.similarity(params)

        trans_graph = deepcopy(graph)

        trans_graph.graph["params"] = params

        for i in range(graph.number_of_nodes()):
            trans_graph.nodes[i]["pos"] = list(
                np.matmul(matrix, trans_graph.nodes[i]["pos"])
            )

        return trans_graph

    def untrans(graph: nx.Graph, params=None, iso=True):
        """
        graph: original graph if matrix is None, else transformed graph
        iso: isomorphic (geometrical is not considered)
        """

        untrans_graph = deepcopy(graph)

        if params:
            untrans_graph = Graph.trans(untrans_graph, params)
        del untrans_graph.graph["params"]
        node_num = untrans_graph.number_of_nodes()

        # If we want the transformed graph to be isomorphic to the original, just randomly change some @pos
        if iso:
            changed_nodes = sample(range(node_num), randint(0, node_num))
            for node in changed_nodes:
                for coor in sample((0, 1, 2), choice((1, 2, 3))):
                    untrans_graph.nodes[node]["pos"][coor] = uniform(0, 100)

        return untrans_graph


def genGraph(num):
    graphs = []
    for i in range(num):
        graph = Graph.gen(int((i + 15) / 5))
        trans_graph = Graph.similarize(graph, ParamGen.random())
        untrans_graph = Graph.untrans(trans_graph)

        graphs.append(
            {
                "nodenum": int((i + 15) / 5),
                "original": nx.node_link_data(graph),
                "similar": nx.node_link_data(trans_graph),
                "asimilar": nx.node_link_data(untrans_graph),
            }
        )

    return graphs


def write_to_json(file, num):
    graphs = genGraph(num)

    with open(file, "w") as f:
        json.dump(graphs, f, indent=2)


def write_to_database(num):
    graphs = genGraph(num)
    graphs_json = json.dumps(graphs)
    graphs_data = json.loads(graphs_json)

    uri = "mongodb+srv://chocomint:chocomint@mintdb.b8hvvyz.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri)
    collection = client["MintDB"]["graphs"]
    collection.insert_many(graphs)
    client.close()

def from_colab_again_2(): pass
