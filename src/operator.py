from random import shuffle
import networkx as nx
import copy
import generator
import numpy as np


def shuffle_graph(G):
  nodes = list(G.nodes(data=True))
  shuffle(nodes)

  # Create a mapping from original node IDs to shuffled node IDs
  id_mapping = {node_id: shuffled_id for shuffled_id, (node_id, _) in enumerate(nodes)}

  # Create a new graph with shuffled node attributes and modified edges
  shuffled_G = nx.Graph()
  shuffled_G.add_nodes_from((shuffled_id, attr) for shuffled_id, (_, attr) in enumerate(nodes))

  shuffled_edges = [(id_mapping[u], id_mapping[v]) for u, v in G.edges()]
  shuffled_G.add_edges_from(shuffled_edges)

  return shuffled_G

def similarize_graph(G, _param=None, shuffle=True, reflect=False):
    if not _param: _param = generator.param()
    rotationMatrix = generator.Matrix.rotation(_param)

    img = copy.deepcopy(G)
    img.graph['param'] = _param

    for _, attr in img.nodes(data=True):
      attr['pos'] = list(_param['scaling']*np.matmul(rotationMatrix, attr['pos']) + _param['translation'])
      if reflect : attr['pos'][0] = -attr['pos'][0]
      attr['pos'] =  list(attr['pos'])

    if shuffle: img = shuffle_graph(img)

    return img