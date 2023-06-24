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

def param(dim=2):
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

class Matrix:
  def __rotationX(_param):
    alpha = _param['rotation']['angles'][0]

    return np.array([[cos(alpha), -sin(alpha), 0],
                  [sin(alpha), cos(alpha), 0],
                  [0, 0, 1]])

  def __rotationY(_param):
    beta = _param['rotation']['angles'][1]
    return np.array([[cos(beta), 0, sin(beta)],
                  [0, 1, 0],
                  [-sin(beta), 0, cos(beta)]])

  def __rotationZ(_param):
    gamma = _param['rotation']['angles'][2]
    return np.array([[1, 0, 1],
                  [0, cos(gamma), -sin(gamma)],
                  [0, sin(gamma), cos(gamma)]])

  def rotation(_param = None):
    if not _param: _param = param()
    if _param['dim'] == 3:
      order = _param['rotation']['order']
      matrices = []

      def appendMatrix(matrices, number):
        if number == 0: matrices.append(Matrix.__rotationX(_param))
        if number == 1: matrices.append(Matrix.__rotationY(_param))
        if number == 2: matrices.append(Matrix.__rotationZ(_param))

      [appendMatrix(matrices, number) for number in order]

      rotationMatrix = np.matmul(matrices[2], (np.matmul(matrices[1], matrices[0])))
    
    if _param['dim'] == 2:
      alpha = _param['rotation']['angle']
      rotationMatrix = np.array([[cos(alpha), -sin(alpha)],
                                 [sin(alpha), cos(alpha)]])
    return rotationMatrix

def graph(nodeNum=None, edgeNum=None, dimension=DIMENSION, labelLimit=LABEL_LIMIT):
  
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