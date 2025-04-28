from neo4j import GraphDatabase
from typing import Dict, List, Optional
import logging

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
