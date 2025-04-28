import logging
import time
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

logger = logging.getLogger(__name__)

class Neo4jKnowledgeGraph:
    _driver = None

    @classmethod
    def get_driver(cls, uri: str, user: str, password: str):
        if cls._driver is None:
            try:
                cls._driver = GraphDatabase.driver(uri, auth=(user, password))
                cls._driver.verify_connectivity()
                logger.info(f"Neo4j driver initialized for URI: {uri}")
            except Exception as e:
                logger.critical(f"Failed to create Neo4j driver: {e}")
                cls._driver = None
                raise
        return cls._driver

    @classmethod
    def close_driver(cls):
        if cls._driver:
            cls._driver.close()
            cls._driver = None
            logger.info("Closed Neo4j driver connection.")

    @classmethod
    def execute_cypher_query(cls, query: str, params: dict = None, retries: int = 2, delay: int = 1):
        driver = cls.get_driver(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        if not driver:
            return {"error": "Neo4j connection not available."}

        for attempt in range(retries):
            try:
                with driver.session() as session:
                    result = session.run(query, params or {})
                    records = [dict(record) for record in result]
                    logger.info(f"Cypher query executed successfully (Attempt {attempt + 1}). Query: '{query}', Records returned: {len(records)}")
                    return records
            except CypherSyntaxError as e:
                error_message = f"Cypher Syntax Error: {e}. Query: '{query}', Parameters: {params}"
                logger.error(error_message)
                return {"error": error_message}
            except Exception as e:
                logger.error(f"Error executing Cypher query (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    return {"error": f"Failed to execute Cypher query after {retries} attempts: {e}"}
        return {"error": "Query execution failed unexpectedly."}
