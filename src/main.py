import generate
from visualize import draw, plot
from infer import similarity_infer
import pprint
import numpy as np

pp = pprint.PrettyPrinter(width=40, compact=True, indent=2)
G, H = generate.generate(8, 4, 2, False)

h1 = generate.modify_positions(H)
h2 = generate.modify_colors(H)
h3 = generate.remove_edges(H)

draw(G)
draw(H)
draw(h1)
draw(h2)
draw(h3)
