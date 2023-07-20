import graph
import visualize
import pprint
from check import check
from transform import Transform
import numpy as np
from math import sqrt, pi
from random import uniform, randint


# for i in range(10, 1000, 10):
#     data = graph.generate(i, randint(int(i / 4), i - 1), 2, True, True)
#     if i % 100 == 0:
#         print(i, "vertices")

data = graph.generate(60, 50, 3, equidistant=True)

print("Check the original graph and its similar partner")
check(data.original, data.similar)
print("\n")

# print("Check the original graph and its similar partner modified some positions")
# check(data.original, data.posision_modified)
# print("\n")

# print("Check the original graph and its similar partner modified some colors")
# check(data.original, data.color_modified)
# print("\n")

# print("Check the original graph and its similar partner removed some edges")
# check(data.original, data.edge_removed)
# print("\n")
