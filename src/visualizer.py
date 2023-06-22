import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np

class Visualizer:
  def graph(graph):
   
    nodes = [data['pos'] for _, data in graph.nodes(data=True)]
    label_pos = {node : np.add(data['pos'], [0,2]) for node, data in graph.nodes(data=True)}
    labels = {node : data['label'] for node, data in graph.nodes(data=True)}
    edges = [(u, v) for u, v in graph.edges() if 'unavailable' not in graph[u][v]]

    fig, ax = plt.subplots()

    nx.draw_networkx_labels(graph, pos=label_pos, labels=labels, font_size=7, ax=ax)
    nx.draw_networkx_nodes(graph, pos=nodes, node_size=20, ax=ax)
    nx.draw_networkx_edges(graph, pos=nodes, edgelist=edges, ax=ax)

    ax.set_aspect('equal')
    plt.show()
  
  def convexhull(graph, color='green', ax=None):
    nodes = [data['pos'] for _, data in graph.nodes(data=True)]

    label_pos = {node: np.add(data['pos'], [0, 2]) for node, data in graph.nodes(data=True)}
    labels = {node: data['label'] for node, data in graph.nodes(data=True)}

    has_ax = True
    if not ax: 
        fig, ax = plt.subplots()
        has_ax = False

    nx.draw_networkx_nodes(graph, pos=nodes, ax=ax, node_size=20)
    nx.draw_networkx_labels(graph, pos=label_pos, labels=labels, font_size=8, ax=ax)
    hull = ConvexHull(nodes)
    hull_points = np.append(hull.vertices, hull.vertices[0])
    hull_nodes = [nodes[i] for i in hull_points]
    ax.fill(*zip(*hull_nodes), color=color, alpha=0.3)

    ax.set_aspect('equal')

    if not has_ax: plt.show()

  def convexhulls(graphs):
    fig, ax = plt.subplots()
    Visualizer.convexhull(graphs[0], 'green', ax)
    Visualizer.convexhull(graphs[1], 'blue', ax)

    plt.show()