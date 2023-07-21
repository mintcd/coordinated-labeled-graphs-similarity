import numpy as np
from get import cloud, bijections
import copy
from itertools import permutations, product
import networkx as nx
from math import factorial
from operations import normalized_clouds
from partition import partition_on_norm, partition_on_distance, partition_on_combination
from validate import validate_bijection


def isomorphism_infer(node_mappings, G: nx.Graph, H: nx.Graph):
    print("Testing {} bijections...".format(len(node_mappings)))
    num = 0
    for mapping in node_mappings:
        print("Bijections {}:".format(num))
        if len(mapping["unsure"]) == 0:
            validated = validate_bijection(mapping["sure"], G, H)
            if type(validated) is not bool:
                return validated
        else:
            result = [mapping["sure"]]
            for unsure_mapping in mapping["unsure"]:
                new_mappings = []
                for mapping in mappings:
                    new_mappings.append(
                        mapping + bijections(unsure_mapping[0], unsure_mapping[1])
                    )
                mappings = new_mappings
        num += 1
    return False


def similarity_infer(G, H):
    print("Translating and scaling...")
    g_cloud, h_cloud = normalized_clouds(cloud(G), cloud(H))

    if g_cloud is not None:
        print("Patitioning on norm...")
        rels = partition_on_norm(g_cloud, h_cloud)

        evaluated = norm_evaluate(rels, g_cloud, h_cloud)

        if evaluated:
            rels = evaluated
        else:
            print("Patitioning on distance...")
            similarities = partition_on_distance(g_cloud, h_cloud, rels)
            rels = []
            for similarity in similarities:
                rels += partition_on_combination(similarity, g_cloud, h_cloud)

    node_mappings = []

    if type(rels) is not bool:
        for rel in rels:
            node_mapping = {"sure": [], "unsure": []}
            for ele in rel:
                g_node = g_cloud[ele[0]]["node"]
                h_node = g_cloud[ele[1]]["node"]
                if len(g_node) == 1:
                    node_mapping["sure"].append((g_node[0], h_node[0]))
                else:
                    node_mapping["unsure"].append((g_node, h_node))

            node_mappings.append(node_mapping)
        print("Similarity inference completed.")
        return node_mappings
    return False


def maximal_linearly_independent(bases):
    result = [[], []]

    for i in range(len(bases[0])):
        new_set = [result[0] + [bases[0][i]], result[1] + [bases[1][i]]]

        if np.linalg.matrix_rank(new_set[0]) > np.linalg.matrix_rank(result[0]):
            result = new_set

    return np.array(result)


def norm_evaluate(rels, g_cloud, h_cloud):
    g_rank = np.linalg.matrix_rank(np.array([g_cloud[vec]["pos"] for vec in g_cloud]))
    h_rank = np.linalg.matrix_rank(np.array([h_cloud[vec]["pos"] for vec in h_cloud]))

    if g_rank != h_rank:
        print("Vector sets of different ranks")
        return False

    # Find corresponding linearly independent sets

    nodes = [[], []]

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
        return [rels]
    return False
