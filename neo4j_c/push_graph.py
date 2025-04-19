# neo4j/.py
from neo4j.config import get_driver
from utils.save_load_graph import load_graph

def push_graph_to_neo4j(G):
    driver = get_driver()
    with driver.session() as session:
        for node, attrs in G.nodes(data=True):
            label = attrs.get("type", "Unknown").capitalize()
            props = {k: v for k, v in attrs.items() if v is not None}
            props_str = ", ".join([f"{k}: ${k}" for k in props])
            query = f"""
            MERGE (n:{label} {{id: $id}})
            SET n += {{{props_str}}}
            """
            session.run(query, id=str(node), **props)

        for source, target, attrs in G.edges(data=True):
            rel_type = attrs.get("type", "RELATED_TO").upper()
            props = {k: v for k, v in attrs.items() if k != "type"}
            props_str = ", ".join([f"{k}: ${k}" for k in props])
            query = f"""
            MATCH (a {{id: $source}}), (b {{id: $target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r += {{{props_str}}}
            """
            session.run(query, source=str(source), target=str(target), **props)

if __name__ == "__main__":
    G = load_graph()
    push_graph_to_neo4j(G)
    print("Graph pushed to Neo4j!")
