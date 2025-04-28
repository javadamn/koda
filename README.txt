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
├── rag/
│   ├── embed_chunks.py         #FAISS embedding + index builder/loader

│   └── chunker.py              
│
├── utils/
│   ├── abundance_mapper.py     #parses and maps microbial abundance
│   ├── data_loader.py          #loads and cleans pairwise CSVs
│   ├── id_mapper.py            #maps strain IDs to names
│   ├── save_load_graph.py      #save/load NetworkX graph
│   └── vmh_client.py           #fetches ::: mocks metabolite metadata
│
├── .env                    
├── main.py                     # Main entry point 
├── requirements.txt       
└── README.md                

