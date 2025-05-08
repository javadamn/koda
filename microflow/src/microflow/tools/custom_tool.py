import json
import os
import pickle
import time
from typing import Dict, List, Optional, Type, Union

from crewai.tools import BaseTool
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from pydantic import BaseModel, Field
import networkx as nx
from dotenv import load_dotenv
from tqdm import tqdm

from .neo4j_utils import create_node, create_relationship

import logging

logger = logging.getLogger(__name__)


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""

    argument: str = Field(..., description="Description of the argument.")


class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."


class Neo4jToolInput(BaseModel):
    operation: str = Field(
        ..., description="The operation to perform on the Neo4j database."
    )
    query_info: str = Field(
        ..., description="The query information to perform on the Neo4j database."
    )


class Neo4jTool(BaseTool):
    name: str = "Neo4j Tool"
    description: str = "Use this tool to query the Neo4j database."

    NEO4J_URI: str = os.getenv("NEO4J_URI")
    NEO4J_USER: str = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD")

    _driver: GraphDatabase = None

    @staticmethod
    def get_driver(reset: bool = False) -> GraphDatabase:
        if Neo4jTool._driver == None or reset:
            try:
                Neo4jTool._driver = GraphDatabase.driver(
                    Neo4jTool.NEO4J_URI,
                    auth=(Neo4jTool.NEO4J_USER, Neo4jTool.NEO4J_PASSWORD),
                )
                Neo4jTool._driver.verify_connectivity()
                logger.info(f"Neo4j driver initialized for URI: {Neo4jTool.NEO4J_URI}")
            except Exception as e:
                logger.error(f"Failed to create Neo4j driver: {e}")
                Neo4jTool._driver = (
                    None  # making sure driver is None if connection fails
                )
                raise  # Re-raise the exception to be caught downstream
        return Neo4jTool._driver

    @staticmethod
    def close_driver():
        if Neo4jTool._driver:
            Neo4jTool._driver.close()
            Neo4jTool._driver = None
            logger.info("Neo4j driver closed")

    def _run(self, operation: str, **kwargs) -> str:
        if operation == "query":
            return self._run_query(**kwargs)
        elif operation == "close_driver":
            self.close_driver()
            return "Neo4j driver closed"
        elif operation == "get_schema":
            return self._run_get_schema()
        elif operation == "load_knowledge":
            return self._run_load_knowledge(**kwargs)
        else:
            return {"error": f"Invalid operation: {operation}"}

    def _run_query(self, query_info: str) -> str:
        try:
            data = json.loads(query_info)
            query = data.get("query")
            params = data.get("params")

            if not query or not isinstance(query, str):
                return {
                    "error": "Invalid input: 'query' field is missing or not a string."
                }
            if params is not None and not isinstance(params, dict):
                return {
                    "error": "Invalid input: 'params' field must be a dictionary if provided."
                }

            logger.info(
                f"Executing Cypher via tool. Query: {query[:100]}..., Params: {params}"
            )
            return Neo4jTool.execute_cypher_query(query, params)
        except json.JSONDecodeError:
            logger.info(
                f"Invalid JSON input to execute_cypher_query_tool: {query_info}"
            )
            return {"error": "Invalid input: Input must be a valid JSON string."}
        except Exception as e:
            logger.error(f"Error in execute_cypher_query_tool wrapper: {e}")
            return {"error": f"Unexpected error processing query input: {e}"}

    def _run_get_schema(self) -> str:
        schema = {"node_labels": [], "relationship_types": [], "properties": {}}
        try:
            labels_result = self.execute_cypher_query(
                "CALL db.labels() YIELD label RETURN collect(label) as labels"
            )
            if (
                isinstance(labels_result, list)
                and labels_result
                and "labels" in labels_result[0]
            ):
                schema["node_labels"] = labels_result[0].get("labels", [])
            elif isinstance(labels_result, dict) and "error" in labels_result:
                logger.error(f"Schema query error (labels): {labels_result['error']}")
                return labels_result  # Propagate DB error

            rel_types_result = self.execute_cypher_query(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as relTypes"
            )
            if (
                isinstance(rel_types_result, list)
                and rel_types_result
                and "relTypes" in rel_types_result[0]
            ):
                schema["relationship_types"] = rel_types_result[0].get("relTypes", [])
            elif isinstance(rel_types_result, dict) and "error" in rel_types_result:
                logger.error(
                    f"Schema query error (relTypes): {rel_types_result['error']}"
                )
                return rel_types_result  # Propagate DB error

            target_labels = ["Microbe", "Metabolite", "Pathway"]
            available_labels = [
                label for label in target_labels if label in schema["node_labels"]
            ]

            for label in available_labels:
                # a note from riddit:: this kind of query might be slow on large graphs, consider sampling or optimized procedures if available
                props_query = f"""
                    MATCH (n:{label})
                    WITH keys(n) AS keys
                    UNWIND keys AS key
                    RETURN collect(distinct key) as properties
                    LIMIT 1
                """
                props_result = self.execute_cypher_query(props_query)
                if (
                    isinstance(props_result, list)
                    and props_result
                    and "properties" in props_result[0]
                ):
                    schema["properties"][label] = props_result[0].get("properties", [])
                elif isinstance(props_result, dict) and "error" in props_result:
                    logger.warning(
                        f"Schema query error (properties for {label}): {props_result['error']}"
                    )

            logger.info(f"Retrieved graph schema: {schema}")
            return schema
        except Exception as e:
            logger.error(f"Failed to retrieve graph schema: {e}")
            return {"error": f"Failed to retrieve schema: {e}"}

    def execute_cypher_query(
        self,
        query: str,
        params: Optional[dict] = None,
        retries: int = 2,
        delay: int = 1,
    ) -> Union[List[Dict], Dict[str, str]]:
        """Executes a Cypher query with retry logic and clearer error reporting."""
        try:
            driver = self.get_driver()
        except Exception as e:
            logger.error(f"Cannot execute query, failed to get Neo4j driver: {e}")
            return {"error": f"Neo4j connection failed: {e}"}

        if not driver:
            return {"error": "Neo4j connection not available."}

        for attempt in range(retries):
            try:
                with driver.session() as session:
                    result = session.run(query, params or {})
                    # list comprehension for cleaner record processing
                    records = [record.data() for record in result]
                    logger.info(
                        f"Cypher query executed successfully (Attempt {attempt + 1}). Query: '{query[:100]}...', Records: {len(records)}"
                    )
                    return records
            except CypherSyntaxError as e:
                error_message = (
                    f"Cypher Syntax Error: {e}. Query: '{query}', Params: {params}"
                )
                logger.error(error_message)
                return {"error": error_message}
            except Exception as e:
                logger.error(
                    f"Error executing Cypher query (attempt {attempt + 1}/{retries}): {e}. Query: '{query[:100]}...', Params: {params}"
                )
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))  # exponential backoff
                else:
                    return {
                        "error": f"Failed to execute Cypher query after {retries} attempts: {e}. Query: '{query[:100]}...'"
                    }
        return {"error": "Query execution failed unexpectedly."}

    @staticmethod
    def _run_load_knowledge(path: str) -> str:
        """Convert a NetworkX graph to a Neo4j graph."""
        with open(path, "rb") as f:
            nx_graph = pickle.load(f)

        driver = Neo4jTool.get_driver()

        with driver.session() as session:
            # clear existing data >> can be removed >> for now i just prefered this one enabled
            session.run("MATCH (n) DETACH DELETE n")

            for node_id in nx_graph.nodes():
                node_data = nx_graph.nodes[node_id]
                if "type" not in node_data:
                    print(f"Skipping node {node_id}: Missing 'type' attribute")
                    continue

                labels = []
                properties = {}

                if node_data["type"] == "microbe":
                    labels = ["Microbe"]
                    properties = {
                        "name": node_id,
                        "abundance": node_data.get("abundance"),
                    }

                elif node_data["type"] == "metabolite":
                    labels = ["Metabolite"]
                    properties = {"name": node_data.get("name", node_id)}

                elif node_data["type"] == "pathway":
                    labels = ["Pathway"]
                    properties = {"name": node_data.get("name", node_id)}

                else:
                    print(
                        f"Skipping node {node_id}: Unknown type '{node_data['type']}'"
                    )
                    continue

                # remove None values from properties
                properties = {k: v for k, v in properties.items() if v is not None}

                # create the node in Neo4j
                session.execute_write(create_node, node_id, labels, properties)

            # create all relationships
            for u, v, data in tqdm(
                nx_graph.edges(data=True), total=nx_graph.number_of_edges()
            ):
                if "type" not in data:
                    print(f"Skipping edge {u}->{v}: Missing 'type' attribute")
                    continue

                # source and target node data
                u_data = nx_graph.nodes[u]
                v_data = nx_graph.nodes[v]

                # labels for source and target nodes
                source_labels = []
                if u_data.get("type") == "microbe":
                    source_labels = ["Microbe"]
                elif u_data.get("type") == "metabolite":
                    source_labels = ["Metabolite"]
                elif u_data.get("type") == "pathway":
                    source_labels = ["Pathway"]

                target_labels = []
                if v_data.get("type") == "microbe":
                    target_labels = ["Microbe"]
                elif v_data.get("type") == "metabolite":
                    target_labels = ["Metabolite"]
                elif v_data.get("type") == "pathway":
                    target_labels = ["Pathway"]

                if source_labels == [] or target_labels == []:
                    print(
                        f"Skipping edge {u}->{v}: Missing 'type' attribute in source or target node"
                    )
                    continue

                rel_type = data["type"].upper().replace(" ", "_").replace("-", "_")
                rel_props = {
                    k: v for k, v in data.items() if k != "type" and v is not None
                }

                # print(
                #     f"Processing edge {u}->{v}, "
                #     f"\nsource_labels: {source_labels}, "
                #     f"\ntarget_labels: {target_labels}, "
                #     f"\nrel_type: {rel_type}, "
                #     f"\nrel_props: {rel_props}"
                # )

                session.execute_write(
                    create_relationship,
                    u,
                    source_labels,
                    v,
                    target_labels,
                    rel_type,
                    rel_props,
                )
