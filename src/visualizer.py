import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import ConvexHull

def graph(graph: nx.Graph, color='green', convexhull=False, ax=None):
  if convexhull:
    nodes = [data['pos'] for _, data in graph.nodes(data=True)]

    label_pos = {node: np.add(data['pos'], [0, 2]) for node, data in graph.nodes(data=True)}
    labels = {node: data['label'] for node, data in graph.nodes(data=True)}

    has_ax = True
    if not ax:
        fig, ax = plt.subplots()
        has_ax = False

    hull = ConvexHull(nodes)
    hull_points = np.append(hull.vertices, hull.vertices[0])
    hull_nodes = [nodes[i] for i in hull_points]
    hull_labels = {node: labels[node] for node in graph.nodes() if node in hull.vertices}


    ax.fill(*zip(*hull_nodes), color=color, alpha=0.3)
    hull_graph = graph.subgraph(hull.vertices)

    hull_node_pos = {node: data['pos'] for node, data in hull_graph.nodes(data=True)}
    hull_lable_pos = {node: np.add(data['pos'], [0, 10]) for node, data in hull_graph.nodes(data=True)}

    nx.draw_networkx_nodes(hull_graph, pos=hull_node_pos, ax=ax, node_size=5, node_color=color)
    nx.draw_networkx_labels(hull_graph, pos=hull_lable_pos, labels=hull_labels, font_size=8, ax=ax)

    ax.set_aspect('equal')

    if not has_ax:
        plt.show()
  else:
    nodes = [data['pos'] for _, data in graph.nodes(data=True)]
    label_pos = {node : np.add(data['pos'], [0,4]) for node, data in graph.nodes(data=True)}
    labels = {node : data['label'] for node, data in graph.nodes(data=True)}
    edges = [(u, v) for u, v in graph.edges() if 'unavailable' not in graph[u][v]]

    fig, ax = plt.subplots()

    nx.draw_networkx_labels(graph, pos=label_pos, labels=labels, font_size=8, ax=ax)
    nx.draw_networkx_nodes(graph, pos=nodes, node_size=10, ax=ax)
    nx.draw_networkx_edges(graph, pos=nodes, edgelist=edges, ax=ax)

    ax.set_aspect('equal')
    plt.show()

def graphs(graphs, convexhull=False):
  fig, ax = plt.subplots()
  colors = ['red', 'green', 'blue', 'orange', 'purple']
  for i, _graph in enumerate(graphs):
    graph(_graph, color=colors[i % len(colors)], ax=ax, convexhull=convexhull)

  plt.show()
