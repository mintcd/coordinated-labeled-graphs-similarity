import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


def draw(G, label=True, coor=False):
    labels = {node: (node, attr["color"]) for node, attr in G.nodes(data=True)}
    nx.draw(
        G,
        with_labels=True,
        labels=labels,
        node_color="lightblue",
        edge_color="gray",
        node_size=20,
    )
    plt.show()


def plot(*graphs):
    fig = None
    ax = None
    dimension = None
    colors = ["red", "blue", "green", "purple", "orange", "cyan"]
    legends = []

    for idx, G in enumerate(graphs):
        node_positions = {node: attr["pos"] for node, attr in G.nodes(data=True)}
        node_positions_array = np.array(list(node_positions.values()))
        if dimension is None:
            dimension = node_positions_array.shape[1]
            if dimension == 2:
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111)
            elif dimension == 3:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111, projection="3d")
            else:
                raise ValueError(
                    "Invalid dimension. Only 2D and 3D plots are supported."
                )
        nx.draw_networkx(
            G,
            pos=node_positions,
            with_labels=False,
            node_size=20,
            node_color="lightblue",
            edge_color="gray",
            font_size=8,
        )
        hull = ConvexHull(node_positions_array)
        hull_vertices = np.append(hull.vertices, hull.vertices[0])
        if dimension == 2:
            plt.plot(
                node_positions_array[hull_vertices, 0],
                node_positions_array[hull_vertices, 1],
                color=colors[idx],
                lw=2,
            )
            plt.fill(
                node_positions_array[hull_vertices, 0],
                node_positions_array[hull_vertices, 1],
                color=colors[idx],
                alpha=0.3,
            )
        elif dimension == 3:
            ax.plot(
                node_positions_array[hull_vertices, 0],
                node_positions_array[hull_vertices, 1],
                node_positions_array[hull_vertices, 2],
                color=colors[idx],
                lw=2,
            )
            ax.fill(
                node_positions_array[hull_vertices, 0],
                node_positions_array[hull_vertices, 1],
                node_positions_array[hull_vertices, 2],
                color=colors[idx],
                alpha=0.3,
            )
        legends.append(G.name)

    plt.legend(legends, fontsize=8)
    if dimension == 2:
        plt.axis("equal")
    elif dimension == 3:
        ax.axis("equal")

    plt.show()
