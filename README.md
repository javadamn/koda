<br>[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.05.27.656377-orange)](https://doi.org/10.1101/2025.05.27.656480)<br>
# KODA: An Agentic Framework for KEGG Orthology-Driven Discovery of Antimicrobial Drug Targets in Gut Microbiome

We introduce **KODA**, a multi-agent framework that combines LLMs with a Neo4j-based knowledge graph to identify potential antimicrobial drug targets in the human gut microbiome. KODA enables natural language querying of microbiome data and generates analytical reports focused on KEGG orthologies (KOs) linked to essential microbial genes.



## System Highlights

- **Agents**: Specialized LLM agents collaborate to process queries, generate Cypher, and create reports  
- **Graph Backend**: A structured Neo4j graph links microbes, genes, pathways, and interactions  
- **Evaluator**: LLM-based reviewers assess the relevance and correctness of outputs

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

## Requirements

Key dependencies include:

- `crewai==0.19.0`
- `langchain==0.1.14`
- `openai==1.22.0`
- `neo4j==5.19.0`
- `sentence-transformers==2.7.0`
- `transformers==4.51.3`  
(See full list in [requirements.txt](requirements.txt))

## Usage

1. Set up and populate the Neo4j microbiome knowledge graph  
2. Add your OpenAI and Neo4j credentials to a `.env` file  
3. Run the main script:

```bash
python main.py
```

4. Enter queries like:  
   `"List essential KOs involved in short-chain fatty acid production"`

## Citation

If you use this repository, please cite:

[**KODA: An Agentic Framework for KEGG Orthology-Driven Discovery of Antimicrobial Drug Targets in Gut Microbiome**  
](https://doi.org/10.1101/2025.05.27.656480)

## License

MIT License – see the [LICENSE](LICENSE) file for details.
