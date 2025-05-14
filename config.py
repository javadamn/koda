import os
import logging
from dotenv import load_dotenv
import sys

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")#"gpt-3.5-turbo")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Javadad6908") 

LOG_FILE_NAME = "graphrag_pipeline.log"
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(name)s | %(filename)s:%(lineno)d] - %(message)s'
LOG_LEVEL = logging.INFO #logging.DEBUG: for more verbose logs from all modules

LLM_TEMPERATURE =0.7

#rremove any existinh handlers from the root logger
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)


logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE_NAME, mode='w'), 
        logging.StreamHandler(sys.stdout) 
    ]
)

logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('crewai').setLevel(logging.INFO) #logging.DEBUG for very verbose crew logs

def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance."""
    return logging.getLogger(name)

logger.info(f"Configuration loaded. Logging to console and '{LOG_FILE_NAME}'. Log level: {logging.getLevelName(LOG_LEVEL)}")

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not set.")
if not NEO4J_URI or not NEO4J_USER or not NEO4J_PASSWORD:
    logger.warning("Neo4j connection details (URI, USER, PASSWORD) may be missing.")