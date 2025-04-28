# main1.py
import os
import json
import pickle
import faiss
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from crewai import Agent, Task, Crew
from crewai.tools import tool

#env. vars
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# print("Loaded OPENAI_API_KEY:", api_key)

#initialize llm
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=api_key)
print("Model in use:", llm.model_name)

# -- Knowledge Graph Handler --
class Neo4jKnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.embedder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
        self.index = faiss.read_index("data/faiss_index.index")
        with open("data/faiss_index_chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.chunks[i] for i in indices[0]]

    def cypher_query(self, query: str, params: Dict = {}) -> List[Dict]:
        with self.driver.session() as session:
            result = session.run(query, params)
            return [dict(record) for record in result]

# Global Neo4j object
kg = Neo4jKnowledgeGraph("bolt://localhost:7687", "neo4j", "Javadad6908")
print(kg.cypher_query("MATCH (n) RETURN n LIMIT 1"))


# TOOL DEFINITIONS
# @tool("Biological Entity Extractor")
# def extract_biological_entities(query: str) -> dict:
#     """Extracts microbes, metabolites, processes and terms from a query."""
#     prompt = f'''
#     Extract biological entities from the following query:
#     "{query}"
#     Return a JSON with keys: microbes, metabolites, processes
#     '''
#     result = llm.invoke(prompt)
#     return json.loads(result.content)
@tool("Biological Entity Extractor")
def extract_biological_entities(query: str) -> dict:
    """Extracts microbes, metabolites, processes and terms from a query, plus related chunks."""
    prompt = f'''
    Extract biological entities from the following query:
    "{query}"
    Return a JSON with keys: microbes, metabolites, processes
    '''
    result = llm.invoke(prompt)
    try:
        parsed = json.loads(result.content)
    except Exception as e:
        print(f"[Entity Extractor] Failed to parse LLM output: {e}")
        return {"error": "Failed to parse biological entities"}

    # Add FAISS semantic enrichment
    related_chunks = kg.semantic_search(query)
    parsed["related_chunks"] = [chunk["text"] for chunk in related_chunks[:3]]

    return parsed



@tool("Flux Analyzer")
def flux_analysis_tool(metabolites: List[str]) -> list:
    """Performs metabolic flux analysis between microbes using provided metabolites."""
    cypher = '''
    MATCH (producer:Microbe)-[prod:PRODUCES]->(met:Metabolite)<-[cons:CONSUMES]-(consumer:Microbe)
    WHERE toLower(met.name) IN $metabolites
    RETURN 
        producer.id AS producer,
        met.name AS metabolite,
        consumer.id AS consumer,
        prod.flux AS production_flux,
        cons.flux AS consumption_flux,
        abs(prod.flux - cons.flux) AS flux_imbalance
    ORDER BY flux_imbalance DESC
    LIMIT 10
    '''
    return kg.cypher_query(cypher, {"metabolites": metabolites})


@tool("Health Effect Mapper")
def health_impact_tool(metabolites: List[str]) -> list:
    """Maps given metabolites to their known health effects using the knowledge graph."""
    cypher = '''
    MATCH (m:Metabolite)-[r:ASSOCIATED_WITH]->(e:HealthEffect)
    WHERE m.name IN $metabolites
    RETURN 
        m.name AS metabolite,
        collect(e.description) AS effects,
        avg(r.evidence_strength) AS evidence_score
    ORDER BY evidence_score DESC
    '''
    return kg.cypher_query(cypher, {"metabolites": metabolites})


@tool("Markdown Report Generator")
def report_generation_tool(findings: Dict) -> str:
    """Generates a Markdown-formatted report from the provided findings dictionary."""
    return f"""
# Microbial Community Interaction Report

## Key Relationships
{findings.get('relationships', 'Not provided.')}

## Metabolic Highlights
{findings.get('metabolic_insights', 'None found.')}

## Hypotheses
{findings.get('hypotheses', 'No hypotheses generated.')}

## Health Implications
{findings.get('health_impacts', 'No health outcomes found.')}

## Recommendations
{findings.get('recommendations', 'Further analysis recommended.')}
"""


# --  Pipeline --
class MicrobialAnalysisPipeline:
    def __init__(self):
        self.agents = {
            "query_interpreter": Agent(
                role="Biological Query Analyst",
                goal="Convert user queries into biological entity structures",
                backstory="Expert in systems biology, skilled in parsing microbiome questions",
                tools=[extract_biological_entities],
                verbose=True
            ),
            "flux_analyst": Agent(
                role="Metabolic Flux Analyst",
                goal="Extract meaningful flux-driven interactions from the graph",
                backstory="Specialist in interpreting metabolic models using graph-based analysis",
                tools=[flux_analysis_tool],
                verbose=True
            ),
            "hypothesis_engine": Agent(
                role="Hypothesis Generator",
                goal="Propose testable ideas about microbial community behavior",
                backstory="Trained in community ecology and emergent systems",
                tools=[],
                verbose=True
            ),
            # "health_impact_analyst": Agent(
            #     role="Health Effect Analyst",
            #     goal="Interpret metabolic outcomes in terms of host impact",
            #     backstory="Knows gut-brain axis and SCFA-related host outcomes",
            #     tools=[health_impact_tool],
            #     verbose=True
            # ),
            "report_synthesizer": Agent(
                role="Report Synthesizer",
                goal="Turn findings into readable, structured biological reports",
                backstory="Microbial science communicator",
                tools=[report_generation_tool],
                verbose=True
            )
        }

    def run_analysis(self, query: str) -> str:
        t1 = Task(
            description="Extract biological entities from the user's query",
            expected_output="A JSON dictionary with microbes, metabolites, processes, and health_terms.",
            agent=self.agents["query_interpreter"]
        )
        t2 = Task(
            description="Analyze flux relationships between microbial producers and consumers",
            expected_output="List of cross-feeding relationships with flux data.",
            agent=self.agents["flux_analyst"],
            context=[t1]
        )
        t3 = Task(
            description="Generate novel hypotheses based on flux relationships",
            expected_output="3-5 biological hypotheses as bullet points.",
            agent=self.agents["hypothesis_engine"],
            context=[t2]
        )
        # t4 = Task(
        #     description="Assess health impacts of the relevant metabolites",
        #     expected_output="Health effects of key metabolites, including any known impact on host physiology.",
        #     agent=self.agents["health_impact_analyst"],
        #     context=[t2]
        # )
        t5 = Task(
            description="Generate a structured Markdown report of all findings",
            expected_output="A complete Markdown report with sections on relationships, hypotheses, and health impacts.",
            agent=self.agents["report_synthesizer"],
            context=[t2, t3]#, t4]
        )
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[t1, t2, t3, t5],
            verbose=True
        )
        return crew.kickoff()

####
# agents=[
#         query_interpreter_agent,
#         flux_analyst_agent,
#         # hypothesis_engine_agent,   <-  disabled
#         # health_impact_analyst_agent,  <-  disabled
#         report_synthesizer_agent,

####

# -- Execution --
if __name__ == "__main__":
    query = "Which microbes could support Thiamine synthesis?"
    pipeline = MicrobialAnalysisPipeline()
    result = pipeline.run_analysis(query)

    print("Report generated.")
    final_output = result if isinstance(result, str) else str(result)

    with open("analysis_report.md", "w") as f:
        f.write(final_output)
