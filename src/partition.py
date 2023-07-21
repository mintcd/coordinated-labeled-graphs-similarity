from get import mutual_distances
import numpy as np
from get import cliques
from itertools import permutations, product


def partition_on_norm(g_cloud, h_cloud):
    rels = []
    results = {"one": [], "many": []}

    g_norms = dict()
    for vec in g_cloud:
        norm = np.linalg.norm(g_cloud[vec]["pos"])
        found = False
        for key in g_norms:
            if np.isclose(norm, key):
                g_norms[key].append(vec)
                found = True
                break
        if not found:
            g_norms[norm] = [vec]

    h_norms = dict()
    for vec in h_cloud:
        norm = np.linalg.norm(h_cloud[vec]["pos"])
        found = False
        for key in h_norms:
            if np.isclose(norm, key):
                h_norms[key].append(vec)
                found = True
                break
        if not found:
            h_norms[norm] = [vec]

    for g_norm in sorted(g_norms.keys()):
        for h_norm in sorted(h_norms.keys()):
            if np.isclose(g_norm, h_norm):
                rels.append((g_norms[g_norm], h_norms[h_norm]))

    for rel in rels:
        if len(rel[0]) != len(rel[1]):
            print("Relations of diffrent numbers of participants")

        else:
            if len(rel[0]) == 1 and len(rel[1]) == 1:
                results["one"].append((rel[0][0], rel[1][0]))
            else:
                results["many"].append((rel[0], rel[1]))

    return results


def partition_on_distance(g_cloud, h_cloud, rels):
    dim = g_cloud[0]["pos"].shape[0]
    result = []
    print("Computing mutual distances...")

    for rel in rels["many"]:
        g_mutual = mutual_distances(rel[0], g_cloud)
        h_mutual = mutual_distances(rel[1], h_cloud)

        for g_key in g_mutual:
            for h_key in h_mutual:
                if np.isclose(g_key, h_key):
                    result.append((g_mutual[g_key], h_mutual[h_key]))
                    break

    result = [(cliques(rel[0], dim + 1), cliques(rel[1], dim + 1)) for rel in result]

    min_list_length = np.inf
    max_ele_length = -np.inf
    min_rel = []
    for rel in result:
        new_max_ele_length = max(len(ele) for ele in rel[0])

        if new_max_ele_length > max_ele_length or (
            new_max_ele_length == max_ele_length and len(rel[0]) < min_list_length
        ):
            min_rel = rel
            max_ele_length = new_max_ele_length
            min_list_length = len(rel[0])

    similarities = tuple(product(min_rel[0], min_rel[1]))

    return similarities


def partition_on_combination(similarity, g_cloud, h_cloud):
    dim = g_cloud[0]["pos"].shape[0]
    result = []
    for per in list(permutations(similarity[1])):
        violate = False
        bases = np.array(
            [
                [g_cloud[similarity[0][i]]["pos"] for i in range(dim)],
                [h_cloud[per[i]]["pos"] for i in range(dim)],
            ]
        )
        combs = {}
        for node in g_cloud:
            comb = tuple(np.linalg.solve(np.transpose(bases[0]), g_cloud[node]["pos"]))
            combs[comb] = [node]

        for node in h_cloud:
            comb = tuple(np.linalg.solve(np.transpose(bases[1]), h_cloud[node]["pos"]))

            found = False
            for key in combs:
                if np.all(np.isclose(key, comb)):
                    combs[key].append(node)
                    found = True
                    break
            if not found:
                # print("H's {} has no combination in G".format(node))
                violate = True
                break

        if not violate:
            result.append(list(combs.values()))
    return result
