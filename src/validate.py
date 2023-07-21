def validate_bijection(rels, G, H):
    rels_dict = {}
    for rel in rels:
        rels_dict[rel[0]] = rel[1]

    for i, attr in G.nodes(data=True):
        if attr["color"] != H.nodes[rels_dict[i]]["color"]:
            print(
                "Invalid: G's {} and H's {} have different colors".format(
                    i, rels_dict[i]
                )
            )
            return False
        for j, _ in G.nodes(data=True):
            if G.has_edge(i, j) and not H.has_edge(rels_dict[i], rels_dict[j]):
                print(
                    "Invalid mapping {}...: G has edge  {} but H does not have edge  {}".format(
                        (i, j), (i, j), (rels_dict[i], rels_dict[j])
                    )
                )
                return False
            elif not G.has_edge(i, j) and H.has_edge(rels_dict[i], rels_dict[j]):
                print(
                    "Invalid mapping {}: G does not have edge{} but H has edge{}".format(
                        (i, j), (i, j), (rels_dict[i], rels_dict[j])
                    )
                )
                return False
    print("Valid similarity {}".format(rels_dict))
    return rels
