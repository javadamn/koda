# tools.py
import json
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field
from crewai.tools import BaseTool # tool decorator is not needed if class-based

from neo4j_handler import Neo4jKnowledgeGraph
import config

logger = config.get_logger(__name__)

# Pydantic model for ExecuteCypherQueryToolClass input
class ExecuteCypherQueryToolSchema(BaseModel):
    """Input schema for ExecuteCypherQueryTool."""
    query: str = Field(description="The Cypher query string to be executed.")
    params: Optional[dict] = Field(default=None, description="An optional dictionary of parameters for the Cypher query.")

# Class-based tool for executing Cypher queries
class ExecuteCypherQueryToolClass(BaseTool):
    name: str = "Execute Cypher Query Tool"
    description: str = (
        "Executes a given Cypher query with parameters against the Neo4j database. "
        "The agent should provide 'query' and 'params' as arguments conforming to the schema."
    )
    args_schema: type[BaseModel] = ExecuteCypherQueryToolSchema

    def _run(self, query: str, params: Optional[dict] = None) -> Union[List[Dict], Dict[str, str]]:
        try:
            if not query:
                 logger.error("Query is empty.")
                 return {"error": "Input 'query' cannot be empty."}
            logger.info(f"Executing Cypher via tool (class-based). Query: {query[:100]}..., Params: {params}")
            return Neo4jKnowledgeGraph.execute_cypher_query(query, params)
        except Exception as e:
            logger.error(f"Error in ExecuteCypherQueryTool _run method: {e}", exc_info=True)
            return {"error": f"Unexpected error processing query input: {e}"}

# --- Class-based Get Graph Schema Tool ---
class GetGraphSchemaToolSchema(BaseModel):
    """Input schema for GetGraphSchemaTool. Accepts no arguments."""
    pass # No arguments needed

class GetGraphSchemaToolClass(BaseTool):
    name: str = "Get Graph Schema Tool"
    description: str = (
        "Retrieves the basic schema of the Neo4j graph, including node labels, relationship types, "
        "and properties for Microbe, Metabolite, Pathway, and KO nodes. "
        "This tool takes no arguments."
    )
    args_schema: type[BaseModel] = GetGraphSchemaToolSchema # Use the empty schema

    def _run(self) -> Union[Dict[str, Union[List[str], Dict[str, List[str]]]], Dict[str, str]]:
        # The agent might still pass an empty dict "{}" as input, but Pydantic with an empty schema should handle it.
        logger.info("GetGraphSchemaToolClass._run() called")
        schema = {
            "node_labels": [],
            "relationship_types": [],
            "properties": {}
        }
        try:
            labels_result = Neo4jKnowledgeGraph.execute_cypher_query("CALL db.labels() YIELD label RETURN collect(label) as labels")
            if isinstance(labels_result, list) and labels_result and "labels" in labels_result[0]:
                schema["node_labels"] = labels_result[0].get("labels", [])
            elif isinstance(labels_result, dict) and "error" in labels_result: return labels_result

            rel_types_result = Neo4jKnowledgeGraph.execute_cypher_query("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as relTypes")
            if isinstance(rel_types_result, list) and rel_types_result and "relTypes" in rel_types_result[0]:
                schema["relationship_types"] = rel_types_result[0].get("relTypes", [])
            elif isinstance(rel_types_result, dict) and "error" in rel_types_result: return rel_types_result

            target_labels_in_code = ["Microbe", "Metabolite", "Pathway", "KO"]
            neo4j_label_map = {
                "Microbe": "microbe", "Metabolite": "metabolite",
                "Pathway": "pathway", "KO": "KO"
            }
            available_neo4j_labels = [neo4j_label_map[label] for label in target_labels_in_code if neo4j_label_map.get(label) in schema["node_labels"]]

            for neo4j_label in available_neo4j_labels:
                 schema_key = next((key for key, value in neo4j_label_map.items() if value == neo4j_label), None)
                 if not schema_key: continue
                 props_query = f"MATCH (n:{neo4j_label}) WHERE n IS NOT NULL WITH keys(n) AS keys UNWIND keys AS key RETURN collect(distinct key) as properties LIMIT 1"
                 props_result = Neo4jKnowledgeGraph.execute_cypher_query(props_query)
                 if isinstance(props_result, list) and props_result and "properties" in props_result[0]:
                     schema["properties"][schema_key] = props_result[0].get("properties", [])
                 elif isinstance(props_result, dict) and "error" in props_result:
                     logger.warning(f"Schema query error (properties for {schema_key} using label {neo4j_label}): {props_result['error']}")
                 else:
                     schema["properties"][schema_key] = []
            logger.info(f"Retrieved final graph schema: {schema}")
            return schema
        except Exception as e:
            logger.error(f"Failed to retrieve graph schema: {e}", exc_info=True)
            return {"error": f"Failed to retrieve schema due to unexpected error: {e}"}