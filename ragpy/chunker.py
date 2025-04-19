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


def graph_to_text_chunks(G):
    """
    Creates text chunks per node with richer biological phrasing.
    """
    chunks = []

    for node, data in G.nodes(data=True):
        node_type = data.get("type", "Unknown").capitalize()
        name = data.get("name", node)
        description = data.get("description", "")

        lines = [f"{node_type} Name: {name}"]
        if description:
            lines.append(f"Description: {description}")

        outgoing = list(G.out_edges(node, data=True))
        incoming = list(G.in_edges(node, data=True))

        if outgoing or incoming:
            lines.append("Biological Relationships:")

        for _, target, edge_data in outgoing:
            rel_type = edge_data.get("type", "RELATED_TO").upper()
            flux = edge_data.get("flux")
            target_name = G.nodes[target].get("name", target)
            rel_desc = edge_data.get("description", "")
            line = f"→ {rel_type} → {target_name}"
            if flux:
                line += f" [Flux: {flux:.2e}]"
            if rel_desc:
                line += f" - {rel_desc}"
            lines.append(line)

        for source, _, edge_data in incoming:
            rel_type = edge_data.get("type", "RELATED_TO").upper()
            flux = edge_data.get("flux")
            source_name = G.nodes[source].get("name", source)
            rel_desc = edge_data.get("description", "")
            line = f"← {rel_type} ← {source_name}"
            if flux:
                line += f" [Flux: {flux:.2e}]"
            if rel_desc:
                line += f" - {rel_desc}"
            lines.append(line)

        chunks.append({
            "id": node,
            "text": "\n".join(lines),
            "type": data.get("type", "unknown")
        })

    return chunks


def graph_to_crossfeeding_chunks(G):
    """
    Builds event-based text chunks from the structure:
    Microbe A --produces--> Metabolite --consumes--> Microbe B
    """
    chunks = []

    for metabolite, data in G.nodes(data=True):
        if data.get("type") != "metabolite":
            continue

        producers = [src for src, _, d in G.in_edges(metabolite, data=True) if d.get("type") == "produces"]
        consumers = [tgt for _, tgt, d in G.out_edges(metabolite, data=True) if d.get("type") == "consumes"]

        for prod in producers:
            for cons in consumers:
                met_name = G.nodes[metabolite].get("name", metabolite)
                prod_name = G.nodes[prod].get("name", prod)
                cons_name = G.nodes[cons].get("name", cons)
                prod_desc = G[prod][metabolite].get("description", "")
                cons_desc = G[metabolite][cons].get("description", "")
                flux_prod = G[prod][metabolite].get("flux")
                flux_cons = G[metabolite][cons].get("flux")

                lines = [
                    f"Cross-feeding Event:",
                    f"- Producer: {prod_name}",
                    f"- Metabolite: {met_name}",
                    f"- Consumer: {cons_name}",
                ]

                if prod_desc:
                    lines.append(f"- Description: {prod_desc}")
                if cons_desc:
                    lines.append(f"- Consumption: {cons_desc}")

                if flux_prod or flux_cons:
                    lines.append(f"- Flux: Production = {flux_prod:.2e} | Consumption = {flux_cons:.2e}")

                chunks.append({
                    "id": f"{prod}__{metabolite}__{cons}",
                    "text": "\n".join(lines),
                    "type": "crossfeeding"
                })

    return chunks
