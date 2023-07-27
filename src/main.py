import graph
import visualize
from check import test, check
from operations import normalized_graph

data = graph.generate(v=10, dim=2, shuffle=False, equidistant=False)
data = graph.load(2, True, 9)


check(data.original, data.similar)
print("\n")
# check(data.original, data.position_modified)
# print("\n")
# check(data.original, data.color_modified)
# print("\n")
# check(data.original, data.edge_removed)
