import numpy as np
from get import get_cloud, get_rank
import copy


def evaluate(rank, one):
    cloud = [item["src"] for item in one]
    cloud_rank = get_rank(cloud)
    if rank > cloud_rank:
        print("Partition on norm did not yield enough mappins")
        return False
    else:
        independent = [one[0]]
        for rel in one:
            rels = independent + [rel]
            new_rank = get_rank([item["src"] for item in rels])
            if new_rank == len(rels):
                independent = rels
            if new_rank == rank:
                return independent
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


def partition_on_norm(g, h):
    rels = {"one": [], "many": []}

    ratios = [x["norm"] / y["norm"] for x, y in zip(g, h)]
    proportional = all(np.isclose(ratio, ratios[0]) for ratio in ratios)

    for node in h:
        node["pos"] *= ratios[0]
        node["norm"] *= ratios[0]

    i = j = 0
    while i < len(g) and j < len(h):
        norm = g[i]["norm"]
        rel = {"src": [g[i]], "des": []}

        i += 1
        while i < len(g) and np.isclose(g[i]["norm"], norm):
            rel["src"].append(g[i])
            i += 1

        while j < len(h) and np.isclose(h[j]["norm"], norm):
            rel["des"].append(h[j])
            j += 1

        if len(rel["src"]) != len(rel["des"]):
            print("Invalid relation")
            return None

        if len(rel["src"]) == 1:
            rels["one"].append({"src": rel["src"][0], "des": rel["des"][0]})
        else:
            rels["many"].append(rel)

    return g, h, rels


def partition_on_distance(g, h, many):
    def mutual_distance(data):
        distances = {}
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                avails = distances.keys()
                pos_i = data[i]["pos"]
                pos_j = data[j]["pos"]
                distance = np.linalg.norm(pos_i - pos_j)

                found = False
                for avail in avails:
                    if np.isclose(distance, avail):
                        distances[avail].append([data[i]["id"], data[j]["id"]])
                        found = True
                        break
                if not found:
                    distances[distance] = [[data[i]["id"], data[j]["id"]]]
        print(distances[list(distances.keys())[0]])

    for rel in many:
        src_dis, des_dis = mutual_distance(rel["src"]), mutual_distance(rel["des"])


def similarity_infer(G, H):
    print("Translating and scaling...\n")
    g, h = normalize(G), normalize(H)
    g_cloud, h_cloud = get_cloud(g), get_cloud(h)

    rank = get_rank(g_cloud, h_cloud)

    print("Patitioning on norm...")
    g_cloud, h_cloud, rels = partition_on_norm(g_cloud, h_cloud)

    independent = evaluate(rank, rels["one"])

    partition_on_distance(g, h, rels["many"])

    print(
        "Found a linearly independent set.\n"
        if independent
        else "Partitioning on distance...\n"
    )

    if independent:
        src_basis = np.array([rel["src"]["pos"] for rel in independent])
        des_basis = np.array([rel["des"]["pos"] for rel in independent])
        print("Source", src_basis)
        print("Destination", des_basis)

        print("\n")

        print("Calculating linear combinations...")
        for node in g_cloud:
            node["comb"] = np.linalg.solve(src_basis.transpose(), node["pos"])

        for node in h_cloud:
            node["comb"] = np.linalg.solve(des_basis.transpose(), node["pos"])

        print("Infer mappings...")

        for many in rels["many"]:
            for src in many["src"]:
                for des in many["des"]:
                    if all(np.isclose(src["comb"], des["comb"])):
                        rels["one"].append({"src": src, "des": des})
                        break

        return [[item["src"]["id"], item["des"]["id"]] for item in rels["one"]]

    return rels
