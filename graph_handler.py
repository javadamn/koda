# graph_handler.py
import logging
import time
from typing import Dict, List, Optional, Union

from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError

import config # Import configuration

logger = logging.getLogger(__name__)

class Neo4jKnowledgeGraph:
    _driver = None # Class level driver

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
                cls._driver = None
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
        driver = cls.get_driver()
        if not driver:
            return {"error": "Neo4j connection not available."}

        for attempt in range(retries):
            try:
                with driver.session() as session:
                    result = session.run(query, params or {})
                    # Consume results fully before session closes
                    records = [dict(record) for record in result]
                    logger.info(f"Cypher query executed successfully (Attempt {attempt + 1}). Query: '{query}', Parameters: {params}, Records returned: {len(records)}")
                    return records
            except CypherSyntaxError as e:
                error_message = f"Cypher Syntax Error: {e}. Query: '{query}', Parameters: {params}"
                logger.error(error_message)
                return {"error": error_message} # Don't retry syntax errors
            except Exception as e:
                # Check for specific transient errors if needed, otherwise retry general exceptions
                logger.error(f"Error executing Cypher query (attempt {attempt + 1}/{retries}): {e}. Query: '{query}', Parameters: {params}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1)) # Exponential backoff
                else:
                    return {"error": f"Failed to execute Cypher query after {retries} attempts: {e}. Query: '{query}'"}
        # Fallback if loop finishes unexpectedly
        return {"error": "Query execution failed unexpectedly."}