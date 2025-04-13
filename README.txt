graphRAG/
│
├── config/
│   └── paths.py                
│
├── data/          
│   ├── pairwise_results.csv
│   ├── anaerobic_strains.csv
│   └── Genra_GEMs/
│       ├── SamplesSpeciesRelativeAbundance_modified_final.csv
│       └── ListofModelsforSpecies.csv
│
├── outputs/                  
│   ├── graph.gpickle           # Saved NetworkX graph
│   ├── chunks.pkl              # Saved text chunks
│   ├── faiss_index.index       # FAISS index
│   └── faiss_index_chunks.pkl  # Metadata for FAISS index
│
├── graphrag/
│   └── graph_builder.py        # Function to build knowledge graph from data
│
├── neo4j/
│   ├── config.py               # Neo4j connection config via .env
│   ├── push_graph.py           # Push graph to Neo4j
│   └── query_graph.py          # Query Neo4j from Cypher
│
├── rag/
│   ├── embed_chunks.py         # FAISS embedding + index builder/loader
│   ├── rag_pipeline.py         # BioRAGPipeline + query logic
│   └── chunker.py              # Converts graph into natural language text chunks
│
├── utils/
│   ├── abundance_mapper.py     # Parses and maps microbial abundance
│   ├── data_loader.py          # Loads and cleans pairwise CSVs
│   ├── id_mapper.py            # Maps strain IDs to names
│   ├── save_load_graph.py      # Save/load NetworkX graph
│   └── vmh_client.py           # Fetches or mocks metabolite metadata
│
├── .env                        # Environment variables (Neo4j creds etc.)
├── main.py                     # Main entry point — builds pipeline step by step
├── requirements.txt            # Dependencies for reproducibility
└── README.md                   # Project description and usage instructions

