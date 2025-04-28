from langchain_openai import ChatOpenAI
import config  
import sys

logger = config.get_logger(__name__)

try:
    llm = ChatOpenAI(model_name=config.LLM_MODEL,
        temperature=0.2, openai_api_key=config.OPENAI_API_KEY)#for more deterministic cypher/analysis>>lower temp
    logger.info(f"Using LLM: {llm.model_name}")
except Exception as e:
    logger.critical(f"Failed to initialize LLM: {e}")
    sys.exit(1)

def get_llm():
    """Returns the initialized LLM instance."""
    return llm