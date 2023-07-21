import graph
import visualize
from check import test, check
from operations import normalized_graph

data = graph.generate(v=10, dim=3, shuffle=False, equidistant=False, write=True)
data = graph.load(3, False, 10)


check(data.original, data.similar)
print("\n")
check(data.original, data.position_modified)
print("\n")
check(data.original, data.color_modified)
print("\n")
check(data.original, data.edge_removed)
