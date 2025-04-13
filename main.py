# main.py

import os
import numpy as np
import pandas as pd
from pathlib import Path

from config.paths import (
    PAIRWISE_PATH, STRAINS_CSV, ABUNDANCE_CSV, MODELS_CSV
)

from utils.data_loader import read_pairwise_data
from utils.abundance_mapper import get_abundance
from utils.vmhcache import VMHCacheClient
from utils.id_mapper import build_id_to_strain_map, replace_ids_with_names
from utils.save_load_graph import save_graph, load_graph
from utils.graph_builder import create_graphrag_knowledge_graph
from ragpy.chunker import graph_to_text_chunks_from_neo4j
from ragpy.embed_chunks import embed_chunks, build_faiss_index, load_faiss_index
from ragpy.rag_pipeline import BioRAGPipeline, query_graphrag

GRAPH_PATH = "outputs/graph.gpickle"
CHUNK_PATH = "outputs/chunks.pkl"
INDEX_PATH = "outputs/faiss_index"

import pickle


def main():
    # Step 1: Load or build the knowledge graph
    if os.path.exists(GRAPH_PATH):
        print("📦 Loading saved graph...")
        G = load_graph(GRAPH_PATH)
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
        save_graph(G, GRAPH_PATH)
        print(f"✅ Graph built and saved with {len(G.nodes)} nodes.")

    # Step 2: Chunk the graph into text
    if os.path.exists(CHUNK_PATH):
        print("📚 Loading chunks from file...")
        with open(CHUNK_PATH, "rb") as f:
            chunks = pickle.load(f)
    else:
        print("🧩 Chunking graph into text...")
        chunks = graph_to_text_chunks_from_neo4j(G)
        with open(CHUNK_PATH, "wb") as f:
            pickle.dump(chunks, f)
        print(f"✅ Chunks saved ({len(chunks)} total).")

    # Step 3: Build or load FAISS index
    index_file = f"{INDEX_PATH}.index"
    chunk_file = f"{INDEX_PATH}_chunks.pkl"

    if os.path.exists(index_file) and os.path.exists(chunk_file):
        print("📈 Loading FAISS index...")
        index, loaded_chunks = load_faiss_index(INDEX_PATH)
    else:
        print("💥 Generating embeddings and building index...")
        embeddings = embed_chunks(chunks)
        build_faiss_index(chunks, embeddings, INDEX_PATH)
        index, loaded_chunks = load_faiss_index(INDEX_PATH)

    # Step 4: Run RAG query
    print("🧠 Initializing BioRAG pipeline...")
    rag_pipeline = BioRAGPipeline()

    query = "Which microbes are responsible for producing butyrate and how does it affect gut health?"
    print(f"❓ Query: {query}")

    context = query_graphrag(query, rag_pipeline.embedding_model, index, loaded_chunks)
    answer = rag_pipeline.generate_answer(context, query)

    print("\n🧬 Final Answer:\n" + answer)


if __name__ == "__main__":
    main()
