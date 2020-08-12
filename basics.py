class Node:
    def __init__(self, attr_name):
        self.attr_name = attr_name
        self.edges = []

    def add_edge(self, e):
        self.edges.append(e)

    def is_leaf(self):
        return len(self.edges) == 0


class Edge:
    def __init__(self, label, to):
        self.label = label
        self.to = to  # node


class Example:
    def __init__(self, features_values, tag):
        self.features_values = features_values  # list
        self.tag = tag  # yes or no


# read examples from a given file , return list of Example objects and list of attributes(first line in file)
def read_examples(file_name):
    with open(file_name) as f:
        content = f.readlines()
    content = [line.strip() for line in content]
    return [Example(ex.split('\t')[:-1], ex.split('\t')[-1]) for ex in content[1:]], content[0].split('\t')[:-1]