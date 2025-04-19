def validate_hypothesis(hypothesis: str):
    """Check hypothesis against graph patterns"""
    validation_query = """
    MATCH (m1:Microbe)-[r1:PRODUCES]->(met:Metabolite)<-[r2:CONSUMES]-(m2:Microbe)
    WHERE NOT (m1)-[:INHIBITS|PROMOTES]->(m2)
    RETURN m1.name, met.name, m2.name, 
           r1.flux/r2.flux as flux_ratio
    ORDER BY abs(flux_ratio - 1) DESC
    """
    results = neo4j.run(validation_query)
    return llm.evaluate_fit(hypothesis, results)


def bio_aware_embedding(chunks):
    """Augment embeddings with biological context"""
    model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
    enriched_chunks = [
        f"{c['text']} [Domain: Microbial Metabolism] [Key Terms: {extract_terms(c['text'])}]"
        for c in chunks
    ]
    return model.encode(enriched_chunks)


def flux_weighted_search(query_embedding, index, chunks):
    """Weight FAISS results by metabolic flux values"""
    distances, indices = index.search(query_embedding, 50)
    weighted_scores = [
        (i, d * (1 + chunk['flux'] if 'flux' in chunk else 1))
        for i, d in zip(indices[0], distances[0])
    ]
    return sorted(weighted_scores, key=lambda x: x[1])[:5]