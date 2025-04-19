# utils/save_load_graph.py
import os
import pickle
import networkx as nx

def save_graph(graph, path="data/graph.gpickle"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)

def load_graph(path="data/graph.gpickle"):
    with open(path, "rb") as f:
        return pickle.load(f)
