import networkx as nx
from random import randint, random, choice, sample, randrange, uniform, shuffle, seed
import numpy as np
from scipy.spatial.transform import Rotation
import json
from math import pi, sin, cos
from itertools import combinations
from copy import deepcopy
from pymongo import MongoClient

SCALE_LIMIT = 5
TRANS_LIMIT = 100
POINT_LIMIT = 100
NODE_LIMIT = 100
DIMENSION = 2
LABEL_LIMIT = 20

class ParamGen:
  @staticmethod
  def random(dim=2):
    param = {}

    param['dim'] = dim
    param['rotation'] = {}
    if dim == 2:
        param['rotation']['angle'] = round(uniform(-pi, pi), 2)
    elif dim == 3:
        param['rotation']['angles'] = [round(uniform(-pi, pi), 2) for _ in range(dim)]
        param['rotation']['order'] = sample([0, 1, 2], 3)

    param['scaling'] = round(uniform(-SCALE_LIMIT, SCALE_LIMIT), 2)
    param['translation'] = [round(uniform(-TRANS_LIMIT, TRANS_LIMIT), 2) for _ in range(dim)]

    return param

class MatrixGen:
  def __rotationX(param):
    alpha = param['rotation']['angles'][0]

    return np.array([[cos(alpha), -sin(alpha), 0],
                  [sin(alpha), cos(alpha), 0],
                  [0, 0, 1]])

  def __rotationY(param):
    beta = param['rotation']['angles'][1]
    return np.array([[cos(beta), 0, sin(beta)],
                  [0, 1, 0],
                  [-sin(beta), 0, cos(beta)]])

  def __rotationZ(param):
    gamma = param['rotation']['angles'][2]
    return np.array([[1, 0, 1],
                  [0, cos(gamma), -sin(gamma)],
                  [0, sin(gamma), cos(gamma)]])

  def rotation(param = None):
    if not param: param = ParamGen.random()
    if param['dim'] == 3:
      order = param['rotation']['order']
      matrices = []

      def appendMatrix(matrices, number):
        if number == 0: matrices.append(MatrixGen.__rotationX(param))
        if number == 1: matrices.append(MatrixGen.__rotationY(param))
        if number == 2: matrices.append(MatrixGen.__rotationZ(param))

      [appendMatrix(matrices, number) for number in order]

      rotationMatrix = np.matmul(matrices[2], (np.matmul(matrices[1], matrices[0])))
    
    if param['dim'] == 2:
      alpha = param['rotation']['angle']
      rotationMatrix = np.array([[cos(alpha), -sin(alpha)],
                                 [sin(alpha), cos(alpha)]])
    return rotationMatrix


class GraphGen:
  def random(nodeNum=None, edgeNum=None, dimension=DIMENSION, labelLimit=LABEL_LIMIT):
    
    G = nx.Graph()
    G.graph['dim'] = dimension

    if not nodeNum: nodeNum = randint(2, NODE_LIMIT)
    if not edgeNum: edgeNum = randint(0, int(nodeNum / 2 * (nodeNum - 1)))

    # Generate nodes
    coordinates = [[randint(-POINT_LIMIT, POINT_LIMIT) for _ in range(dimension)] for _ in range(nodeNum)]
    G.add_nodes_from([(i, {'label': choice(range(labelLimit)),
                            'pos': coordinates[i]}) for i in range(nodeNum)])

    # Generate edges
    G.add_edges_from(sample(list(combinations(range(nodeNum), 2)), edgeNum))

    return G
  
  def similarism(graph, param=None):

    if not param: param = ParamGen.random()
    rotationMatrix = MatrixGen.rotation(param)

    img = deepcopy(graph)
    img.graph['param'] = param

    for _, attr in img.nodes(data=True):
      attr['pos'] = list(param['scaling']*np.matmul(rotationMatrix, attr['pos']) + param['translation'])

    return img

  def inversed_similarism(graph, param=None):

      if not param: param = ParamGen.random()
      rotationMatrix = MatrixGen.rotation(param)

      img = deepcopy(graph)
      img.graph['param'] = param

      for _, attr in img.nodes(data=True):
        attr['pos'] = list(param['scaling']*np.matmul(rotationMatrix, attr['pos']) + param['translation'])    
        attr['pos'][0] = -attr['pos'][0]

      return img