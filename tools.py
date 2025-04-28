# tools.py
import json
from typing import Dict, List, Optional, Union
from crewai.tools import tool
from neo4j_handler import Neo4jKnowledgeGraph 
import config 

logger = config.get_logger(__name__)

# tools
@tool("Execute Cypher Query Tool")
def execute_cypher_query_tool(query_info: str) -> Union[List[Dict], Dict[str, str]]:
    """
    Executes a given Cypher query with parameters against the Neo4j database.
    Input must be a JSON string containing 'query' (string) and 'params' (dict, optional).
    Returns the list of result records (dictionaries) or an error dictionary.
    Example input: '{"query": "MATCH (m:Microbe {name: $name}) RETURN m.abundance", "params": {"name": "E. coli"}}'
    """
    try:
        data = json.loads(query_info)
        query = data.get("query")
        params = data.get("params")

        if not query or not isinstance(query, str):
            return {"error": "Invalid input: 'query' field is missing or not a string."}
        if params is not None and not isinstance(params, dict):
             return {"error": "Invalid input: 'params' field must be a dictionary if provided."}

        logger.info(f"Executing Cypher via tool. Query: {query[:100]}..., Params: {params}")
        return Neo4jKnowledgeGraph.execute_cypher_query(query, params)
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON input to execute_cypher_query_tool: {query_info}")
        return {"error": "Invalid input: Input must be a valid JSON string."}
    except Exception as e:
        logger.error(f"Error in execute_cypher_query_tool wrapper: {e}")
        return {"error": f"Unexpected error processing query input: {e}"}

@tool("Get Graph Schema Tool")
def get_graph_schema_tool(_) -> Union[Dict[str, Union[List[str], Dict[str, List[str]]]], Dict[str, str]]:
    """
    Retrieves the basic schema of the Neo4j graph, including node labels, relationship types,
    and properties for Microbe, Metabolite, and Pathway nodes.
    Input is ignored (can be empty string or None).
    Returns a dictionary describing the schema or an error dictionary.
    """
    schema = {
        "node_labels": [],
        "relationship_types": [],
        "properties": {}
    }
    try:
        labels_result = Neo4jKnowledgeGraph.execute_cypher_query("CALL db.labels() YIELD label RETURN collect(label) as labels")
        if isinstance(labels_result, list) and labels_result and "labels" in labels_result[0]:
            schema["node_labels"] = labels_result[0].get("labels", [])
        elif isinstance(labels_result, dict) and "error" in labels_result:
             logger.error(f"Schema query error (labels): {labels_result['error']}")
             return labels_result # Propagate DB error

        rel_types_result = Neo4jKnowledgeGraph.execute_cypher_query("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as relTypes")
        if isinstance(rel_types_result, list) and rel_types_result and "relTypes" in rel_types_result[0]:
            schema["relationship_types"] = rel_types_result[0].get("relTypes", [])
        elif isinstance(rel_types_result, dict) and "error" in rel_types_result:
             logger.error(f"Schema query error (relTypes): {rel_types_result['error']}")
             return rel_types_result # Propagate DB error

        target_labels = ["Microbe", "Metabolite", "Pathway"]
        available_labels = [label for label in target_labels if label in schema["node_labels"]]

        for label in available_labels:
             #a note from riddit:: this kind of query might be slow on large graphs, consider sampling or optimized procedures if available
            props_query = f"""
                MATCH (n:{label})
                WITH keys(n) AS keys
                UNWIND keys AS key
                RETURN collect(distinct key) as properties
                LIMIT 1
            """
            props_result = Neo4jKnowledgeGraph.execute_cypher_query(props_query)
            if isinstance(props_result, list) and props_result and "properties" in props_result[0]:
                schema["properties"][label] = props_result[0].get("properties", [])
            elif isinstance(props_result, dict) and "error" in props_result:
                logger.warning(f"Schema query error (properties for {label}): {props_result['error']}")

        logger.info(f"Retrieved graph schema: {schema}")
        return schema
    except Exception as e:
        logger.error(f"Failed to retrieve graph schema: {e}")
        return {"error": f"Failed to retrieve schema: {e}"}