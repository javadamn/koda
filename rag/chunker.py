# rag/.py
def graph_to_text_chunks_from_neo4j(driver):
    with driver.session() as session:
        result = session.run("""
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.id AS id, n.type AS type, n.name AS name, n.description AS description,
               COLLECT({type: type(r), description: r.description, target: m.id}) AS interactions
        """)

        chunks = []
        for record in result:
            interactions = "\n".join([
                f"{r['type']} → {r['target']} ({r.get('description', '')})"
                for r in record["interactions"] if r['type'] is not None
            ])
            text = f"{record['type'].capitalize()} Node: {record['name'] or record['id']}\n{record.get('description') or ''}\nInteractions:\n{interactions}"
            chunks.append({
                "id": record["id"],
                "text": text,
                "type": record["type"]
            })
    return chunks
