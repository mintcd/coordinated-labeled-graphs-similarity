import numpy as np
import copy
from itertools import combinations


def get_rank(*clouds):
    rank = None

    for idx, cloud in enumerate(clouds):
        pos = np.array([node["pos"] for node in cloud])
        if not rank:
            rank = np.linalg.matrix_rank(pos)
        elif rank != np.linalg.matrix_rank(pos):
            print("Two graphs has sets of nodes of different ranks")
            return None

    return rank


def get_adjacency(G):
    return [(item, list(value.keys())) for item, value in G.adjacency()]


def get_cloud(G):
    id = 0
    cloud = {}
    for node, attr in G.nodes(data=True):
        found = False
        for key in cloud.keys():
            if np.all(np.isclose(attr["pos"], cloud[key]["pos"])):
                cloud[key]["node"].append(node)
                found = True
                break
        if not found:
            cloud[id] = {"node": [node], "pos": attr["pos"]}
            id += 1

    return cloud


def get_centroid(cloud):
    pass


def get_cliques(pairs, k):
    cliques = copy.deepcopy(pairs)
    result = copy.deepcopy(pairs)

    for i in range(3, k + 1):
        done = set()
        groups = {}
        for clique in cliques:
            if clique[0 : i - 2] in groups.keys():
                groups[clique[0 : i - 2]] += clique[i - 2 :]
            else:
                groups[clique[0 : i - 2]] = clique[i - 2 :]

        clique_plus = []

        for item in groups:
            groups[item] = [pair for pair in combinations(groups[item], 2)]
            for value in groups[item]:
                if value in pairs:
                    clique_plus.append(item + value)
                    done.update(item + value)

        cliques = clique_plus

        result_copy = copy.deepcopy(result)

        for old_clique in result_copy:
            for ele in old_clique:
                if ele in done:
                    result.remove(old_clique)
                    break

        result += clique_plus

    return result
