import logging
import time
from config import OPENAI_API_KEY
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

from langchain_openai import ChatOpenAI
from neo4j_handler import Neo4jKnowledgeGraph
from agents import MicrobialAnalysisPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_llm():
    try:
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY)
        logger.info(f"Using LLM: {llm.model_name}")
        return llm
    except Exception as e:
        logger.critical(f"Failed to initialize LLM: {e}")
        exit(1)

if __name__ == "__main__":
    try:
        Neo4jKnowledgeGraph.get_driver(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

        llm = initialize_llm()
        pipeline = MicrobialAnalysisPipeline(llm)

        query = "What microbes both produce and consume Thiamine? What is the net flux if possible?"
        logger.info(f"Running analysis for query: {query}")
        result = pipeline.run_analysis(query)
        logger.info(f"Analysis result: {result}")
    except Exception as e:
        logger.critical(f"An error occurred during setup or execution: {e}")
    finally:
        Neo4jKnowledgeGraph.close_driver()
