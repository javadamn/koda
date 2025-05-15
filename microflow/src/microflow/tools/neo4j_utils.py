def create_node(tx, node_id, labels, properties):
    # let MERGE node by name and set properties
    query = f"MERGE (n:{':'.join(labels)} {{name: $name}}) " "SET n += $props"
    tx.run(query, name=node_id, props=properties)


def create_relationship(
    tx,
    source_name,
    source_labels,
    target_name,
    target_labels,
    rel_type,
    rel_props,
):
    # MATCH nodes and CREATE relationship with properties
    query = (
        f"MATCH (a:{':'.join(source_labels)} {{name: $source_name}}), "
        f"(b:{':'.join(target_labels)} {{name: $target_name}}) "
        f"CREATE (a)-[r:{rel_type}]->(b) "
        "SET r += $props"
    )
    tx.run(query, source_name=source_name, target_name=target_name, props=rel_props)


def _get_node_labels(node_data: dict) -> list[str]:
    """Helper function to get Neo4j labels for a node."""
    node_type = node_data.get("type")
    if node_type == "microbe":
        return ["Microbe"]
    elif node_type == "metabolite":
        return ["Metabolite"]
    elif node_type == "pathway":
        return ["Pathway"]
    elif node_type == "kegg_orthology":
        return ["KO"]
    else:
        return []
