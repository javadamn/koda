# main.py
import os
import json
import pickle
import numpy as np
import faiss
from typing import Dict, List
from crewai import Agent, Task, Process
from langchain.agents import Tool
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI  # ✅
from crewai import Crew

from neo4j import GraphDatabase
# from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer


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


# -- Agents Setup --
class MicrobialAnalysisAgents:
    def __init__(self, neo4j_kg: Neo4jKnowledgeGraph):
        self.neo4j = neo4j_kg
        self.llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)
        print("✅ Model in use:", self.llm.model_name)

        # Define tools
        self.entity_tool = Tool(
            name="Biological Entity Extractor",
            func=self._extract_biological_entities,
            description="Extracts microbial strains, metabolites, processes, and health terms from a natural language query."
        )

        self.flux_tool = Tool(
            name="Flux Analyzer",
            func=self._flux_analysis_tool,
            description="Analyzes cross-feeding flux balance relationships between microbes."
        )

        self.hypothesis_tool = Tool(
            name="Hypothesis Generator",
            func=self._pattern_discovery_tool,
            description="Generates new microbial interaction hypotheses."
        )

        self.health_tool = Tool(
            name="Health Effect Mapper",
            func=self._health_impact_tool,
            description="Maps metabolites to their known health effects."
        )

        self.report_tool = Tool(
            name="Markdown Report Generator",
            func=self._report_generation_tool,
            description="Generates a Markdown report based on structured biological findings."
        )

        # Define agents
        self.agents = {
            "query_interpreter": Agent(
                role="Biological Query Analyst",
                goal="Convert user queries into biological entity structures",
                backstory="Expert in systems biology, skilled in parsing microbiome questions",
                tools=[self.entity_tool],
                verbose=True
            ),
            "flux_analyst": Agent(
                role="Metabolic Flux Analyst",
                goal="Extract meaningful flux-driven interactions from the graph",
                backstory="Specialist in interpreting metabolic models using graph-based analysis",
                tools=[self.flux_tool],
                verbose=True
            ),
            "hypothesis_engine": Agent(
                role="Hypothesis Generator",
                goal="Propose testable ideas about microbial community behavior",
                backstory="Trained in community ecology and emergent systems",
                tools=[self.hypothesis_tool],
                verbose=True
            ),
            "health_impact_analyst": Agent(
                role="Health Effect Analyst",
                goal="Interpret metabolic outcomes in terms of host impact",
                backstory="Knows gut-brain axis and SCFA-related host outcomes",
                tools=[self.health_tool],
                verbose=True
            ),
            "report_synthesizer": Agent(
                role="Report Synthesizer",
                goal="Turn findings into readable, structured biological reports",
                backstory="Microbial science communicator",
                tools=[self.report_tool],
                verbose=True
            ),
        }

    # ----- Tool implementations -----
    def _extract_biological_entities(self, query: str) -> Dict:
        prompt = f"""
        Extract biological entities from the following query:
        "{query}"
        Return a JSON with keys: microbes, metabolites, processes, health_terms
        """
        print("🧠 Inside entity extraction, using model:", self.llm.model_name)

        result = self.llm.invoke(prompt)
        return json.loads(result.content)

    def _flux_analysis_tool(self, params: Dict) -> List[Dict]:
        metabolites = params.get("metabolites", [])
        if not metabolites:
            return []
        cypher = """
        MATCH (producer:Microbe)-[prod:PRODUCES]->(met:Metabolite)<-[cons:CONSUMES]-(consumer:Microbe)
        WHERE met.name IN $metabolites
        RETURN 
            producer.name AS producer,
            met.name AS metabolite,
            consumer.name AS consumer,
            prod.flux AS production_flux,
            cons.flux AS consumption_flux,
            abs(prod.flux - cons.flux) AS flux_imbalance
        ORDER BY flux_imbalance DESC
        LIMIT 10
        """
        return self.neo4j.cypher_query(cypher, {"metabolites": metabolites})

    def _pattern_discovery_tool(self, context: str) -> str:
        prompt = f"""
        Based on the following microbial relationships:
        {context}

        Propose 3 hypotheses about:
        - Cross-feeding interactions
        - Community stability
        - Metabolic bottlenecks
        - Dietary interventions
        """
        return self.llm.invoke(prompt).content

    def _health_impact_tool(self, metabolites: List[str]) -> List[Dict]:
        cypher = """
        MATCH (m:Metabolite)-[r:ASSOCIATED_WITH]->(e:HealthEffect)
        WHERE m.name IN $metabolites
        RETURN 
            m.name AS metabolite,
            collect(e.description) AS effects,
            avg(r.evidence_strength) AS evidence_score
        ORDER BY evidence_score DESC
        """
        return self.neo4j.cypher_query(cypher, {"metabolites": metabolites})

    def _report_generation_tool(self, findings: Dict) -> str:
        template = f"""
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
        return template


# -- Pipeline Runner --
class MicrobialAnalysisPipeline:
    def __init__(self, neo4j_kg: Neo4jKnowledgeGraph):
        self.agents = MicrobialAnalysisAgents(neo4j_kg).agents

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
        t4 = Task(
            description="Assess health impacts of the relevant metabolites",
            expected_output="Health effects of key metabolites, including any known impact on host physiology.",
            agent=self.agents["health_impact_analyst"],
            context=[t2]
        )
        t5 = Task(
            description="Generate a structured Markdown report of all findings",
            expected_output="A complete Markdown report with sections on relationships, hypotheses, and health impacts.",
            agent=self.agents["report_synthesizer"],
            context=[t2, t3, t4]
        )

        # process = Process([t1, t2, t3, t4, t5])#tasks

        crew = Crew(
            agents=[t1.agent, t2.agent, t3.agent, t4.agent, t5.agent],
            tasks=[t1, t2, t3, t4, t5],
            verbose=True
)


        return crew.kickoff()




# -- Main Run --
if __name__ == "__main__":
    # Initialize Neo4j Graph
    kg = Neo4jKnowledgeGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="your_password"
    )

    query = "Which lactate-producing microbes could support butyrate synthesis in a low-fiber diet?"

    pipeline = MicrobialAnalysisPipeline(kg)
    report = pipeline.run_analysis(query)
    print("✅ Report generated.")
    print("Report length:", len(str(report)))

    print("\n🔬 Final Report:\n")
    if isinstance(report, str):
        print(report)
    else:
        print("✅ Report generated.")

    with open("analysis_report.md", "w") as f:
        f.write(report)
