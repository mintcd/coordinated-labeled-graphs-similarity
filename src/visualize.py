import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull


def draw(*graph_list):
    num_graphs = len(graph_list)

    if num_graphs == 1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Original Graph")
        G = graph_list[0]
        color_mapping = {}
        labels = {}
        color_list = []
        node_colors = [attr["color"] for _, attr in G.nodes(data=True)]
        for color in node_colors:
            if color not in color_mapping:
                color_mapping[color] = len(color_mapping)
            color_list.append(color_mapping[color])
        labels.update({node: node for node, _ in G.nodes(data=True)})
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos=pos,
            with_labels=True,
            node_color=color_list,
            cmap=plt.cm.tab10,
            edge_color="gray",
            node_size=100,
            font_size=8,
            ax=ax,
        )
    else:
        fig, axes = plt.subplots(1, num_graphs, figsize=(4 * num_graphs, 4))
        color_mapping = {}
        labels = {}
        for i, G in enumerate(graph_list):
            ax = axes[i]
            color_list = []
            node_colors = [attr["color"] for _, attr in G.nodes(data=True)]
            for color in node_colors:
                if color not in color_mapping:
                    color_mapping[color] = len(color_mapping)
                color_list.append(color_mapping[color])
            labels.update({node: node for node, _ in G.nodes(data=True)})
            pos = nx.spring_layout(G)
            if i == 0:
                ax.set_title("Original Graph")
            else:
                ax.set_title("Similar Graph")
            nx.draw(
                G,
                pos=pos,
                with_labels=True,
                node_color=color_list,
                cmap=plt.cm.tab10,
                edge_color="gray",
                node_size=100,
                font_size=8,
                ax=ax,
            )

    plt.tight_layout()
    plt.show()


def plot(*graphs, label=False):
    fig = plt.figure()
    dimension = graphs[0].graph["dim"]
    if dimension == 2:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection="3d")
    colors = ["red", "blue", "green", "purple", "orange", "cyan"]

    for idx, G in enumerate(graphs):
        node_positions = {node: attr["pos"] for node, attr in G.nodes(data=True)}
        node_positions_array = np.array(list(node_positions.values()))

        if label:
            # Offset the y-coordinate for node labels to make them appear higher
            y_offset = 0.1  # Adjust this value to control the offset
            for node, (x, y) in node_positions.items():
                ax.text(x, y + y_offset, node, ha="center", va="center", fontsize=8)

        if dimension == 2:
            nx.draw_networkx(
                G,
                pos=node_positions,
                with_labels=False,
                node_size=20,
                node_color=colors[idx],
                edge_color="gray",
                font_size=8,
                ax=ax,
            )

            hull = ConvexHull(node_positions_array)
            hull_vertices = np.append(hull.vertices, hull.vertices[0])
            hull_positions = node_positions_array[hull_vertices]

            ax.plot(
                hull_positions[:, 0],
                hull_positions[:, 1],
                color=colors[idx],
                lw=2,
            )
            ax.fill(
                hull_positions[:, 0],
                hull_positions[:, 1],
                color=colors[idx],
                alpha=0.3,
            )

            ax.axis("equal")

        elif dimension == 3:
            ax.scatter(
                node_positions_array[:, 0],
                node_positions_array[:, 1],
                node_positions_array[:, 2],
                c=colors[idx],
                marker="o",
            )

            hull = ConvexHull(node_positions_array)
            hull_vertices = node_positions_array[hull.vertices]

            # Triangulate the hull vertices for plotting trisurf
            hull_tri = ConvexHull(hull_vertices)
            ax.plot_trisurf(
                hull_vertices[:, 0],
                hull_vertices[:, 1],
                hull_vertices[:, 2],
                triangles=hull_tri.simplices,
                alpha=0.3,
                color=colors[idx],
            )

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_box_aspect((1, 1, 1))
    plt.show()
