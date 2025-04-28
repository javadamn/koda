import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from neo4j import GraphDatabase
from crewai import Agent, Task, Crew
from crewai.tools import tool
import logging

# --- Configuration ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Javadad6908")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LLM Initialization ---
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, openai_api_key=OPENAI_API_KEY)
logger.info(f"Using LLM: {llm.model_name}")

# --- Knowledge Graph Handler ---
class Neo4jKnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def cypher_query(self, query: str, params: Dict = {}) -> List[Dict]:
        """Executes a Cypher query and returns the results as a list of dictionaries."""
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                records = [dict(record) for record in result]
                logger.info(f"Cypher query executed successfully. Query: '{query}', Parameters: {params}")
                return records
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}. Query: '{query}', Parameters: {params}")
            return []

    def get_related_microbes(self, metabolite_name: str) -> List[str]:
        """Finds microbes that produce or consume a given metabolite."""
        query = """
            MATCH (met:Metabolite {name: $metabolite})
            MATCH (producer:Microbe)-[:PRODUCES]->(met)
            MATCH (consumer:Microbe)<-[:CONSUMES]-(met)
            RETURN DISTINCT producer.name AS producer, consumer.name AS consumer,
                            producer.flux AS production_flux, consumer.flux AS consumption_flux
        """
        results = self.cypher_query(query, {"metabolite": metabolite_name})
        producers = list(set([row['producer'] for row in results]))
        consumers = list(set([row['consumer'] for row in results]))

        logger.info(f"Found producers: {producers}, consumers: {consumers} for metabolite: {metabolite_name}")
        return producers, consumers

    def get_metabolite_info(self, metabolite_name: str) -> Optional[Dict]:
        """Retrieves information about a metabolite, including health effects."""
        query = """
            MATCH (m:Metabolite {name: $metabolite})
            OPTIONAL MATCH (m)-[r:ASSOCIATED_WITH]->(e:HealthEffect)
            RETURN
                m.name AS metabolite,
                m.description AS metabolite_description,
                collect(e.description) AS health_effects,
                avg(r.evidence_strength) AS evidence_score
            """
        results = self.cypher_query(query, {"metabolite": metabolite_name})
        if results:
            logger.info(f"Metabolite info found: {results[0]}")
            return results[0]
        else:
            logger.warning(f"No metabolite info found for: {metabolite_name}")
            return None

# --- Global Neo4j object ---
try:
    kg = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    logger.info("Connected to Neo4j.")
except Exception as e:
    logger.critical(f"Failed to connect to Neo4j: {e}. The application cannot run without a database connection.")
    kg = None

# --- Tool Definitions ---
@tool("Analyze Microbial Interactions")
def analyze_microbial_interactions(metabolite_name: str) -> dict:
    """
    Analyzes the interactions of microbes related to a specific metabolite in the community.
    It finds producers and consumers of the metabolite and retrieves its potential health effects.
    """
    if kg is None:
        return {"error": "No connection to the knowledge graph."}

    producers, consumers = kg.get_related_microbes(metabolite_name)
    metabolite_info = kg.get_metabolite_info(metabolite_name)

    interaction_summary = ""
    if producers and consumers:
        interaction_summary = f"In this community, the following microbes interact through {metabolite_name}:\n\n"
        producer_names = ", ".join(producers)
        consumer_names = ", ".join(consumers)
        interaction_summary += f"* Producers: {producer_names}\n* Consumers: {consumer_names}\n"
    elif producers:
        interaction_summary = f"In this community, the following microbes produce {metabolite_name}: {', '.join(producers)}.\n"
    elif consumers:
        interaction_summary = f"In this community, the following microbes consume {metabolite_name}: {', '.join(consumers)}.\n"
    else:
        interaction_summary = f"No direct interactions for {metabolite_name} found in this community."

    hypotheses = ""
    if producers and consumers:
        hypotheses = (
            f"Potential hypotheses regarding {metabolite_name} interactions:\n"
            f"* The producers may play a key role in providing {metabolite_name} for the growth of the consumers.\n"
            f"* The balance of {metabolite_name} production and consumption could be a critical factor in community stability."
        )
    else:
        hypotheses = f"No specific hypotheses related to {metabolite_name} could be generated."

    health_effects_text = ""
    if metabolite_info and metabolite_info['health_effects']:
        health_effects_text = f"{metabolite_name} has the following potential health implications: {', '.join(metabolite_info['health_effects'])}."
    else:
        health_effects_text = f"No direct health implications found for {metabolite_name} in the knowledge graph."

    report = (
        f"{interaction_summary}\n\n"
        f"{hypotheses}\n\n"
        f"{health_effects_text}\n"
    )
    return {"report": report}



# --- Pipeline ---
class MicrobialAnalysisPipeline:
    def __init__(self):
        self.agents = {
            "analyzer": Agent(
                role="Microbial Community Analyst",
                goal="Analyze microbial community interactions and generate a report based on the user's query.",
                backstory="Expert in microbial ecology, proficient in using a knowledge graph to understand microbe-metabolite relationships and their implications.",
                tools=[analyze_microbial_interactions],
                verbose=True
            ),
            "reporter": Agent( #added
                role="Report Writer",
                goal="Generate a concise and informative report.",
                backstory="Experienced scientific communicator, skilled at summarizing complex information.",
                tools=[],
                verbose=True
            )
        }

    def run_analysis(self, query: str) -> str:
        """Runs the microbial community analysis pipeline for a given user query."""
        analyzer_agent = self.agents["analyzer"]
        reporter_agent = self.agents["reporter"] #added

        task_analysis = Task(
            description=f"Analyze the microbial community based on the user query: '{query}'.  Focus on identifying key microbial interactions and their potential implications.  The output should be a dictionary.",
            expected_output="A dictionary containing a report summarizing the analysis.",
            agent=analyzer_agent
        )

        task_report = Task( #added
            description="Generate a concise report of the analysis.",
            expected_output="A short, informative report.",
            agent=reporter_agent,
            context=[task_analysis] #takes the output of the analysis task
        )

        crew = Crew(
            agents=[analyzer_agent, reporter_agent], #added reporter_agent
            tasks=[task_analysis, task_report], #added task_report
            verbose=True
        )
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            error_message = f"Error running the crew: {e}"
            logger.error(error_message)
            return error_message



# --- Execution ---
if __name__ == "__main__":
    if kg is None:
        print("Exiting due to database connection error.")
        exit(1)

    query = "Which microbes are likely to be involved in Thiamine metabolism and what are the potential consequences of this interaction?"
    pipeline = MicrobialAnalysisPipeline()
    result = pipeline.run_analysis(query)

    print("Report generated.")
    final_output = result if isinstance(result, str) else str(result)

    with open("analysis_report.md", "w") as f:
        f.write(final_output)

    print(f"Report saved to analysis_report.md")


