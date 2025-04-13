import networkx as nx

def save_graph(graph, path="outputs/graph.gpickle"):
    nx.write_gpickle(graph, path)

def load_graph(path="outputs/graph.gpickle"):
    return nx.read_gpickle(path)