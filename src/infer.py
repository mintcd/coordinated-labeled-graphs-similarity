import numpy as np
from get import get_cloud, get_rank, get_cliques
import copy
from itertools import permutations
import networkx as nx
from math import factorial


def isomorphism_infer(rels, G: nx.Graph, H: nx.Graph):
    print("Similarity inference completed.")
    number = 1
    new_rels = {"one": [], "many": []}
    for rel in rels:
        if len(rel[0]) == 1:
            new_rels["one"].append((rel[0][0], rel[1][0]))
        else:
            new_rels["many"].append(rel)
            number *= factorial(len(rel[0]))

    if len(new_rels["many"]) > 0:
        print(
            "{} unsure mappings of same-position nodes: {}".format(
                number, new_rels["many"][0:9]
            )
        )
    else:
        print("All mappings:", new_rels["one"][0:9], "...")
        print("Validate the bijection...")
        validate_bijection(new_rels["one"], G, H)


def validate_bijection(rels, G: nx.Graph, H: nx.Graph):
    rels_dict = {}
    for rel in rels:
        rels_dict[rel[0]] = rel[1]

    for i, attr in G.nodes(data=True):
        if attr["color"] != H.nodes[rels_dict[i]]["color"]:
            print(
                "Invalid: G's {} and H's {} have different colors".format(
                    i, rels_dict[i]
                )
            )
            return False
        for j, _ in G.nodes(data=True):
            if G.has_edge(i, j) and not H.has_edge(rels_dict[i], rels_dict[j]):
                print(
                    "Invalid: G has edge {} but H does not have edge {}".format(
                        (i, j), (rels_dict[i], rels_dict[j])
                    )
                )
                return False
            elif not G.has_edge(i, j) and H.has_edge(rels_dict[i], rels_dict[j]):
                print(
                    "Invalid: G does not have edge{} but H has edge{}".format(
                        (i, j), (rels_dict[i], rels_dict[j])
                    )
                )
                return False
    print("Two graphs are similar")
    return True


def similarity_infer(G, H):
    print("Translating and scaling...")

    g_cloud, h_cloud = get_cloud(G), get_cloud(H)

    g_cloud, h_cloud = normalized_cloud(g_cloud, h_cloud)

    if g_cloud is not None:
        print("Patitioning on norm...")
        rels = partition_on_norm(g_cloud, h_cloud)

        evaluated = evaluate(rels, g_cloud, h_cloud)

        if evaluated:
            rels = evaluated
        else:
            print("Patitioning on distance...")
            rels = partition_on_distance(g_cloud, h_cloud, rels)

        if type(rels) is not bool:
            result = []
            for rel in rels:
                result.append((g_cloud[rel[0]]["node"], h_cloud[rel[1]]["node"]))
        return result
    else:
        return False


def maximal_linearly_independent(bases):
    result = [[], []]

    for i in range(len(bases[0])):
        new_set = [result[0] + [bases[0][i]], result[1] + [bases[1][i]]]

        if np.linalg.matrix_rank(new_set[0]) > np.linalg.matrix_rank(result[0]):
            result = new_set

    return np.array(result)


def validate_bases(rels, g_cloud, h_cloud):
    g_combs = {}
    h_combs = {}

    rank = np.linalg.matrix_rank(np.array([g_cloud[vec]["pos"] for vec in g_cloud]))
    basis_rank = np.linalg.matrix_rank(rels[0])

    bases = []

    if basis_rank == rank:
        bases = maximal_linearly_independent(rels)

    result = []

    for id in g_cloud:
        comb = np.linalg.solve(np.transpose(bases[0]), g_cloud[id]["pos"])
        g_combs[tuple(comb)] = id

    for id in h_cloud:
        comb = np.linalg.solve(np.transpose(bases[1]), h_cloud[id]["pos"])
        h_combs[tuple(comb)] = id

    for g_key in g_combs.keys():
        for h_key in h_combs.keys():
            if np.all(np.isclose(g_key, h_key)):
                result.append((g_combs[g_key], h_combs[h_key]))

    if len(result) < len(g_combs):
        return False

    return result


def evaluate(rels, g_cloud, h_cloud):
    g_rank = np.linalg.matrix_rank(np.array([g_cloud[vec]["pos"] for vec in g_cloud]))
    h_rank = np.linalg.matrix_rank(np.array([h_cloud[vec]["pos"] for vec in h_cloud]))

    if g_rank != h_rank:
        print("Vector sets of different ranks")
        return False

    # Find corresponding linearly independent sets

    pos = [[], []]
    found = False

    for rel in rels["one"]:
        g_pos = pos[0] + [g_cloud[rel[0]]["pos"]]
        h_pos = pos[1] + [h_cloud[rel[1]]["pos"]]

        new_g_rank = np.linalg.matrix_rank(g_pos)

        if new_g_rank > np.linalg.matrix_rank(pos[0]):
            pos[0] = g_pos
            pos[1] = h_pos
        if g_rank == new_g_rank:
            found = True
            break

    if found:
        print(
            "Found bases {} of G and {} of H".format(np.array(pos[0]), np.array(pos[1]))
        )
        print("Computing linear combinations...")

        g_combs = {}
        h_combs = {}

        for id in g_cloud:
            comb = np.linalg.solve(np.transpose(pos[0]), g_cloud[id]["pos"])
            g_combs[tuple(comb)] = id

        for id in h_cloud:
            comb = np.linalg.solve(np.transpose(pos[1]), h_cloud[id]["pos"])
            h_combs[tuple(comb)] = id

        rels = []

        for g_key in g_combs.keys():
            for h_key in h_combs.keys():
                if np.all(np.isclose(g_key, h_key)):
                    rels.append((g_combs[g_key], h_combs[h_key]))

        return rels
    else:
        return False


def normalize(G):
    g = copy.deepcopy(G)
    cloud = list(
        map(lambda item: {"id": item[0], "pos": item[1]["pos"]}, G.nodes(data=True))
    )
    centroid = sum([item["pos"] for item in cloud]) / G.number_of_nodes()
    for node, attr in g.nodes(data=True):
        attr["pos"] -= centroid
        attr["norm"] = np.linalg.norm(attr["pos"])
    return g


def normalized_cloud(g_cloud, h_cloud):
    # Translate to O
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
    result = {}

    def mutual_distance(ids, cloud):
        positions = np.array([cloud[point]["pos"] for point in cloud])
        pos_i = positions[ids]
        pos_j = positions[ids][:, np.newaxis]
        diffs = pos_j - pos_i
        distances = np.linalg.norm(diffs, axis=2)

        distances = np.triu(distances, k=1)
        unique_distances = np.unique(distances)

        result = {}
        for distance in unique_distances:
            pairs = np.argwhere(np.isclose(distances, distance))
            result[distance] = [(ids[pair[0]], ids[pair[1]]) for pair in pairs]

        return result

    result = []

    print("Computing mutual distances...")

    for rel in rels["many"]:
        g_mutual = mutual_distance(rel[0], g_cloud)
        h_mutual = mutual_distance(rel[1], h_cloud)

        for g_key in g_mutual.keys():
            if np.isclose(g_key, 0.0):
                continue

            if g_key in h_mutual:
                h_key = g_key
                result.append((g_mutual[g_key], h_mutual[h_key]))

    new_result = []

    print("Get cliques...")

    for pair in result:
        new_pair = []
        for ele in pair:
            new_pair.append(get_cliques(ele, dim + 1))
        new_result.append(new_pair)

    result = new_result

    min_length = np.inf
    max_ele = -np.inf
    min_rel = []
    for rel in result:
        longest_ele = -np.inf
        for ele in rel[0]:
            longest_ele = max(len(ele), longest_ele)
        if longest_ele >= max(max_ele, dim) and len(rel[0]) < min_length:
            min_rel = rel
            max_ele = len(ele)
            min_length = len(rel[0])

    similarities = []

    for des in min_rel[1]:
        combs = permutations(des)
        for comb in combs:
            similarities.append((min_rel[0][0], comb))

    print("Checking {} potential mappings".format(len(similarities)))
    for similarity in similarities:
        bases = np.array(
            [
                [g_cloud[id]["pos"] for id in similarity[0]],
                [h_cloud[id]["pos"] for id in similarity[1]],
            ]
        )
        rels = validate_bases(bases, g_cloud, h_cloud)
        if type(rels) is not bool:
            print("Valid mapping: {}.".format(similarity))
            return rels
        else:
            # print("Invalid mapping: {}. Try another one...".format(similarity))
            pass
    return False
