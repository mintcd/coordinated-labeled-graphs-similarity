import graph
import visualize
from check import check


data = graph.generate(v=21, c=2, dim=2, equidistant=True)

visualize.plot(data.original, data.similar)

print("Check the original graph and its similar partner")
check(data.original, data.similar)
print("\n")

print("Check the original graph and its similar partner modified some positions")
check(data.original, data.posision_modified)
print("\n")

print("Check the original graph and its similar partner modified some colors")
check(data.original, data.color_modified)
print("\n")

print("Check the original graph and its similar partner removed some edges")
check(data.original, data.edge_removed)
print("\n")
