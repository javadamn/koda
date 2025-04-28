import json
from neo4j_handler import Neo4jKnowledgeGraph

# Tool for executing the Cypher query
def execute_cypher_query_tool(query_info: str):
    try:
        data = json.loads(query_info)
        query = data.get("query")
        params = data.get("params")

        if not query or not isinstance(query, str):
            return {"error": "Invalid input: 'query' field is missing or not a string."}
        if params is not None and not isinstance(params, dict):
            return {"error": "Invalid input: 'params' field must be a dictionary if provided."}

        return Neo4jKnowledgeGraph.execute_cypher_query(query, params)
    except json.JSONDecodeError:
        return {"error": "Invalid input: Input must be a valid JSON string."}
    except Exception as e:
        return {"error": f"Unexpected error processing query input: {e}"}
