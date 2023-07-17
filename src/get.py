import numpy as np


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
    cloud = []
    for node, attr in G.nodes(data=True):
        avail = list(
            filter(lambda item: np.all(np.isclose(attr["pos"], item["pos"])), cloud)
        )
        if len(avail) == 1:
            avail[0]["id"].append(node)
        else:
            cloud.append(
                {"id": [node], "pos": attr["pos"], "norm": np.linalg.norm(attr["pos"])}
            )

    return cloud


def get_centroid(cloud):
    pass
