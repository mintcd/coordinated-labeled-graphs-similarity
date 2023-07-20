import networkx as nx
from infer import similarity_infer, isomorphism_infer


def check(G, H):
    rels = similarity_infer(G, H)
    if type(rels) is not bool:
        isomorphism_infer(rels, G, H)
