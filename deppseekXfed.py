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
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7688")
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
                logger.info(f"Cypher query executed successfully. Query: '{query}', Parameters: {params}, Records: {records}")
                return records
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}. Query: '{query}', Parameters: {params}")
            return []

    def get_schema(self) -> str:
        """Retrieves a simplified schema of the Neo4j graph, including node labels, relationship types, and properties."""
        query = """
            CALL apoc.meta.graph()
            YIELD nodes, relationships
            UNWIND nodes AS node
            UNWIND relationships AS relationship
            RETURN
                collect(DISTINCT node.label) AS nodeLabels,
                collect(DISTINCT relationship.type) AS relationshipTypes,
                collect(DISTINCT keys(relationship)) AS relationshipProperties
        """
        try:
            results = self.cypher_query(query)
            if results:
                schema_info = results[0]
                node_labels = schema_info.get("nodeLabels", [])
                relationship_types = schema_info.get("relationshipTypes", [])
                relationship_properties = schema_info.get("relationshipProperties", [])
                schema_string = f"Node Labels: {', '.join(node_labels)}\nRelationship Types: {', '.join(relationship_types)}\nRelationship Properties: {relationship_properties}"  # Simplified schema
                return schema_string
            else:
                return "Could not retrieve graph schema."
        except Exception as e:
            logger.error(f"Error retrieving graph schema: {e}")
            return "Error retrieving graph schema."

# --- Global Neo4j object ---
try:
    kg = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    logger.info("Connected to Neo4j.")
except Exception as e:
    logger.critical(f"Failed to connect to Neo4j: {e}. The application cannot run without a database connection.")
    kg = None

# --- Tool Definitions ---
@tool("Generate and Execute Cypher Query")
def generate_and_execute_cypher(query: str, schema: str) -> List[Dict]:
    """
    Generates a Cypher query based on the user's natural language query and the provided knowledge graph schema,
    and then executes the query against the Neo4j database.
    """
    if kg is None:
        return [{"role": "knowledge graph", "content": "No connection to the knowledge graph."}]

    prompt = f"""
    You are a Cypher query writer.
    Your task is to translate the user's natural language query into a valid Cypher query that can be executed
    on a knowledge graph. You are provided with the schema of the graph to help you write the query.

    Graph Schema:
    {schema}

    User Query:
    {query}

    Respond only with the Cypher query. Do not include any explanations or other text.
    """
    try:
        cypher_query = llm.invoke(prompt).content
        logger.info(f"Generated Cypher query: {cypher_query}")
    except Exception as e:
        error_message = f"Error generating Cypher query: {e}"
        logger.error(error_message)
        return [{"role": "error", "content": error_message}]

    results = kg.cypher_query(cypher_query)
    if not results:
        return [{"role": "knowledge graph", "content": "No results found for the generated Cypher query."}]
    return results  # Returns the raw data from the KG



@tool("Analyze and Report from Graph Data")
def analyze_and_report_from_graph_data(query: str, knowledge_graph_data: List[Dict]) -> str:
    """
    Analyzes the data retrieved from the knowledge graph and generates a concise report in Markdown format,
    tailored to the user's query.  This tool focuses on interpreting the raw data from the graph.
    """
    if not knowledge_graph_data:
        return "There was no information to analyze."

    report = f"**Analysis Report for Query: {query}**\n\n"
    report += "Here's a summary of the information found in the knowledge graph:\n\n"

    #  This is the core logic that needs to be adapted based on the query and the
    #  structure of the data returned from the knowledge graph.
    #  The current implementation is a placeholder and needs to be expanded.

    first_record = knowledge_graph_data[0]
    if not first_record:
        return "No information found in the knowledge graph."

    # Adapt report generation based on the structure of the first record.  This is crucial
    if "metabolite_name" in first_record:
        report += "### Metabolite Interactions:\n"
        for record in knowledge_graph_data:
            report += f"-   Metabolite: {record['metabolite_name']}\n"
            report += f"    -   Description: {record['metabolite_description'] or 'Not available'}\n"
            report += f"    -   Producers: {record['producer'] or 'None'}\n"
            report += f"    -   Consumers: {record['consumer'] or 'None'}\n"
            report += f"    -   Production Flux: {record['production_flux'] or 'Not available'}\n"
            report += f"    -   Consumption Flux: {record['consumption_flux'] or 'Not available'}\n"
    elif "microbe_name" in first_record:
        report += "### Microbe Relationships:\n"
        for record in knowledge_graph_data:
            report += f"-   Microbe: {record['microbe_name']}\n"
            report += f"    -   Species: {record['species'] or 'Not available'}\n"
            report += f"    -   Related Node: {record['related_node_name'] or 'None'}\n"
            report += f"    -   Relationship Type: {record['relationship_type'] or 'None'}\n"
    elif "subsystem_name" in first_record:
        report += "### Pathway Analysis:\n"
        for record in knowledge_graph_data:
            report += f"-   Subsystem: {record['subsystem_name']}\n"
            report += f"    -   Description: {record['subsystem_description'] or 'Not available'}\n"
            report += f"    -   Microbe: {record['microbe_name'] or 'None'}\n"
            report += f"    -   Relationship Type: {record['relationship_type'] or 'None'}\n"
            report += f"    -   Subsystem Score: {record['subsystem_score'] or 'Not available'}\n"
    else:
        report += "### General Graph Overview:\n"
        for record in knowledge_graph_data:
            report += f"-   Node 1: {record['node1_name']}\n"
            report += f"    -   Relationship Type: {record['relationship_type']}\n"
            report += f"    -   Node 2: {record['node2_name']}\n"

    report += "\n\nThis report provides a summary of the knowledge graph data relevant to your query. Further analysis and interpretation may be needed."
    return report

# --- Pipeline ---
class MicrobialAnalysisPipeline:
    def __init__(self):
        self.agents = {
            "query_agent": Agent(
                role="Knowledge Graph Query Generator",
                goal="Understand the user's query and generate an appropriate Cypher query to retrieve information from the knowledge graph.",
                backstory="Expert in translating natural language queries into Cypher graph queries.  Understands the structure and data within the microbial community knowledge graph.",
                tools=[generate_and_execute_cypher],
                verbose=True
            ),
            "reporter_agent": Agent(
                role="Microbial Community Analyst and Report Writer",
                goal="Analyze the data retrieved from the knowledge graph and generate a concise and informative report for a biologist.",
                backstory="Experienced scientific communicator with a strong understanding of microbial communities and their interactions.  Skilled at summarizing complex data into readable reports.",
                tools=[analyze_and_report_from_graph_data],
                verbose=True
            ),
        }

    def run_analysis(self, query: str) -> str:
        """Runs the microbial community analysis pipeline for a given user query."""
        query_agent = self.agents["query_agent"]
        report_agent = self.agents["reporter_agent"]

        graph_schema = kg.get_schema() # Get the schema
        

        task_get_data = Task(
            description=f"Generate a Cypher query to retrieve data from the knowledge graph that is relevant to the user's query: '{query}'.  Use the provided graph schema to help you construct the query. Return the results of the query.",
            expected_output="A list of dictionaries representing the relevant data from the knowledge graph.",
            agent=query_agent,
            context=[]#[graph_schema]
        )

        task_report = Task(
            description="Analyze the knowledge graph data and generate a concise report in Markdown format, tailored to the user's query.",
            expected_output="A short, informative report in Markdown format.",
            agent=report_agent,
            context=[task_get_data]
        )

        crew = Crew(
            agents=[query_agent, report_agent],
            tasks=[task_get_data, task_report],
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
