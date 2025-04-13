# .py
from utils.save_load_graph import save_graph
from graphrag import create_graphrag_knowledge_graph, read_pairwise_data, get_abundance  # etc.

def main():
    # 1. Load and process data
    pairwise_data, strain_mean_biomass, microbial_abundance, metabolite_info = preprocess_data()

    # 2. Build graph
    G = create_graphrag_knowledge_graph(pairwise_data, strain_mean_biomass, microbial_abundance, metabolite_info)

    # 3. Save graph
    save_graph(G, "outputs/graph.gpickle")

    print("✅ Graph created and saved!")

if __name__ == "__main__":
    main()
