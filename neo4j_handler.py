import time
from typing import Dict, List, Optional, Union
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
import config 

logger = config.get_logger(__name__)

class Neo4jKnowledgeGraph:
    """Handles interactions with the Neo4j database."""
    _driver = None 

    @classmethod
    def get_driver(cls):
        """Gets a Neo4j driver instance, reusing if possible."""
        if cls._driver is None:
            try:
                cls._driver = GraphDatabase.driver(
                    config.NEO4J_URI,
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
                )
                cls._driver.verify_connectivity()
                logger.info(f"Neo4j driver initialized for URI: {config.NEO4J_URI}")
            except Exception as e:
                logger.critical(f"Failed to create Neo4j driver: {e}")
                cls._driver = None #making sure driver is None ::>> if connection fails
                raise 
        return cls._driver

    @classmethod
    def close_driver(cls):
        """Closes the Neo4j driver connection."""
        if cls._driver:
            cls._driver.close()
            cls._driver = None
            logger.info("Closed Neo4j driver connection.")

    @classmethod
    def execute_cypher_query(cls, query: str, params: Optional[dict] = None, retries: int = 2, delay: int = 1) -> Union[List[Dict], Dict[str, str]]:
        """Executes a Cypher query with retry logic and clearer error reporting."""
        try:
            driver = cls.get_driver()
        except Exception as e:
            logger.error(f"Cannot execute query, failed to get Neo4j driver: {e}")
            return {"error": f"Neo4j connection failed: {e}"}

        if not driver:
            return {"error": "Neo4j connection not available."}

        for attempt in range(retries):
            try:
                with driver.session() as session:
                    result = session.run(query, params or {})
                    #list comprehension for cleaner record processing
                    records = [record.data() for record in result]
                    logger.info(f"Cypher query executed successfully (Attempt {attempt + 1}). Query: '{query[:100]}...', Records: {len(records)}")
                    return records
            except CypherSyntaxError as e:
                error_message = f"Cypher Syntax Error: {e}. Query: '{query}', Params: {params}"
                logger.error(error_message)
                return {"error": error_message}
            except Exception as e:
                logger.error(f"Error executing Cypher query (attempt {attempt + 1}/{retries}): {e}. Query: '{query[:100]}...', Params: {params}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1)) #exponential backoff
                else:
                    return {"error": f"Failed to execute Cypher query after {retries} attempts: {e}. Query: '{query[:100]}...'"}
        return {"error": "Query execution failed unexpectedly."}