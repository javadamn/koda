# main.py

import os
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle
from config.paths import (
    PAIRWISE_PATH, STRAINS_CSV, ABUNDANCE_CSV, MODELS_CSV
)

from utils.data_loader import read_pairwise_data
from utils.abundance_mapper import get_abundance
from utils.vmhcache import VMHCacheClient
from utils.id_mapper import build_id_to_strain_map, replace_ids_with_names
from utils.save_load_graph import save_graph, load_graph
from utils.graph_builder import create_graphrag_knowledge_graph
from ragpy.chunker import graph_to_text_chunks_from_neo4j, graph_to_text_chunks, graph_to_crossfeeding_chunks
from ragpy.embed_chunks import embed_chunks, build_faiss_index, load_faiss_index
from ragpy.rag_pipeline import BioRAGPipeline, query_graphrag
from neo4j_c.configNeo4j import get_driver
import networkx as nx
from collections import Counter

GRAPH_PATH = "data"
CHUNK_PATH = "data/chunks.pkl"
INDEX_PATH = "data/faiss_index"

EMBEDDING_MODEL_NAME = 'pritamdeka/S-PubMedBert-MS-MARCO' #'all-MiniLM-L6-v2'




def main():
    #  1:: loading/build the knowledge graph-------------------------------------------
    if os.path.exists(GRAPH_PATH):
        print("📦 Loading saved graph...")
        G = load_graph()
        #testing the graph
        print(f"Total nodes in G: {len(G.nodes)}")
        print(f"Total edges in G: {len(G.edges)}")
        node_types = Counter(nx.get_node_attributes(G, "type").values())
        print("Node types:", node_types)

        edge_types = Counter([G[u][v]['type'] for u, v in G.edges])
        print("Edge types:", edge_types)

        microbes = [n for n, d in G.nodes(data=True) if d['type'] == 'microbe']
        metabolites = [n for n, d in G.nodes(data=True) if d['type'] == 'metabolite']

        print("Example microbes:")
        for m in microbes[:3]:
            print(m, G.nodes[m])

        print("\nExample metabolites:")
        for m in metabolites[:3]:
            print(m, G.nodes[m])

        example_microbe = microbes[0]
        print(f"\nEdges for {example_microbe}:")
        for neighbor in G.successors(example_microbe):
            print(f"{example_microbe} → {neighbor}, edge type: {G[example_microbe][neighbor]['type']}, description: {G[example_microbe][neighbor].get('description')}")

        example_met = metabolites[0]
        print(f"\nConsumers of {example_met}:")
        for pred in G.predecessors(example_met):
            print(f"{pred} → {example_met}, edge type: {G[pred][example_met]['type']}, flux: {G[pred][example_met].get('flux')}")
        # Stop
        
    else:
        print("🔍 Reading and processing input data...")
        pairwise_data, _, _, _, strain_mean_biomass = read_pairwise_data(PAIRWISE_PATH)

        vmh_client = VMHCacheClient()
        list_EXrxns = sorted({met for pair in pairwise_data for met in pair["crossfed metabolites"]})
        list_EXrxns = [rxn[3:] for rxn in list_EXrxns]
        metabolite_info = vmh_client.get_met_info(list_EXrxns)

        microbial_abundance, xml_models_df = get_abundance(ABUNDANCE_CSV, MODELS_CSV, STRAINS_CSV)

        id_to_strain = build_id_to_strain_map(xml_models_df)
        pairwise_data = replace_ids_with_names(pairwise_data, id_to_strain)

        microbial_abundance.columns = microbial_abundance.columns.astype(str)
        strain_mean_biomass_df = pd.DataFrame([strain_mean_biomass]).rename(columns=str)
        common_cols = microbial_abundance.columns.intersection(strain_mean_biomass_df.columns)
        microbial_abundance = microbial_abundance[common_cols]
        strain_mean_biomass_df = strain_mean_biomass_df[common_cols]
        strain_mean_biomass = {
            id_to_strain[int(k)]: v for k, v in strain_mean_biomass.items() if int(k) in id_to_strain
        }

        G = create_graphrag_knowledge_graph(pairwise_data, strain_mean_biomass, microbial_abundance, metabolite_info)
        save_graph(G)
        print(f"✅ Graph built and saved with {len(G.nodes)} nodes.")

    #2:: chunk the graph into text----------------------------------------------------
    if os.path.exists(CHUNK_PATH):
        print("📚 Loading chunks from file...")
        with open(CHUNK_PATH, "rb") as f:
            chunks = pickle.load(f)
    else:
        print("🔌 Connecting to Neo4j and chunking graph...")
        driver = get_driver()
        print("🧩 Chunking graph into text freom driver...")
        chunks = graph_to_text_chunks_from_neo4j(driver)
        node_chunks = graph_to_text_chunks(G)
        event_chunks = graph_to_crossfeeding_chunks(G)
        chunks=event_chunks
        with open(CHUNK_PATH, "wb") as f:
            pickle.dump(chunks, f)
        print(f"✅ Chunks saved ({len(chunks)} total).")

    # 3:: Building/loading FAISS index-------------------------------------------------
    index_file = f"{INDEX_PATH}.index"
    chunk_file = f"{INDEX_PATH}_chunks.pkl"


    if os.path.exists(index_file) and os.path.exists(chunk_file):
        print("📈 Loading FAISS index...")
        index, loaded_chunks = load_faiss_index(INDEX_PATH)
    else:
        print("💥 Generating embeddings and building index...")
        embeddings = embed_chunks(chunks, model_name=EMBEDDING_MODEL_NAME)
        build_faiss_index(chunks, embeddings, INDEX_PATH)
        index, loaded_chunks = load_faiss_index(INDEX_PATH)

    expected_dim = SentenceTransformer(EMBEDDING_MODEL_NAME).get_sentence_embedding_dimension()
    if index.d != expected_dim:
        print(f"⚠️ FAISS index dimension ({index.d}) doesn't match embedding model dimension ({expected_dim}).")
        print("🔄 Rebuilding FAISS index...")
        embeddings = SentenceTransformer(EMBEDDING_MODEL_NAME).encode([c["text"] for c in chunks], convert_to_numpy=True)
        build_faiss_index(chunks, embeddings, INDEX_PATH)
        index, loaded_chunks = load_faiss_index(INDEX_PATH)

    #4:: run RAG query-----------------------------------------------------
    print("🧠 Initializing BioRAG pipeline...")
    rag_pipeline = BioRAGPipeline(embedding_model_name=EMBEDDING_MODEL_NAME)

    query = "Which microbes are responsible for producing butyrate and how does it affect gut health?"
    print(f"❓ Query: {query}")

    context = query_graphrag(query, rag_pipeline.embedding_model, index, loaded_chunks)
    answer = rag_pipeline.generate_answer(context, query)

    print("\n🧬 Final Answer:\n" + answer)


if __name__ == "__main__":
    main()
