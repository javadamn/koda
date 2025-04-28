import os
import logging
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")#"gpt-3.5-turbo")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Javadad6908") 

#logging if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING) 
logging.getLogger('crewai').setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance."""
    return logging.getLogger(name)

logger.info("Configuration loaded.")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not set.")
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    logger.warning("Neo4j connection details (URI, USER, PASSWORD) may be missing.")