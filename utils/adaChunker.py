def adaptive_chunking(query: str):
    """Dynamically adjust chunking strategy based on query type"""
    chunk_strategies = {
        "relationship": graph_to_crossfeeding_chunks,
        "microbe-centric": graph_to_text_chunks,
        "metabolite-flow": lambda G: [
            f"{s} → {t} via {m} (flux: {f})" 
            for s, t, m, f in get_flux_paths(G)
        ]
    }
    strategy = llm.classify_query_type(query)
    return chunk_strategies[strategy](G)