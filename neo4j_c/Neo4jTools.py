class Neo4jConnector:
    def __init__(self):
        self.driver = get_driver()  # From your existing config
    
    def query_graph(self, query: str) -> str:
        """Handle biological relationship queries"""
        cypher_template = """
        MATCH (producer:Microbe)-[r1:PRODUCES]->(m:Metabolite)<-[r2:CONSUMES]-(consumer:Microbe)
        WHERE toLower(producer.name) CONTAINS toLower($keyword) 
           OR toLower(m.name) CONTAINS toLower($keyword)
        RETURN producer.name, m.name, consumer.name, r1.flux as production, r2.flux as consumption
        ORDER BY r1.flux DESC
        LIMIT 10
        """
        params = self._extract_keywords(query)
        return self._run_cypher(cypher_template, params)
    
    def advanced_cypher(self, params: dict) -> str:
        """Execute complex metabolic pattern queries"""
        flux_query = """
        MATCH (m1:Microbe)-[r1:PRODUCES]->(met:Metabolite)<-[r2:CONSUMES]-(m2:Microbe)
        WITH met, sum(r1.flux) as total_production, sum(r2.flux) as total_consumption
        WHERE total_production > $threshold OR total_consumption > $threshold
        RETURN met.name, total_production, total_consumption,
               abs(total_production - total_consumption) as imbalance
        ORDER BY imbalance DESC
        """
        return self._run_cypher(flux_query, {'threshold': 1e-5})
    
    def _extract_keywords(self, query: str) -> dict:
        """Use LLM to extract biological entities"""
        prompt = f"""
        Extract microbial and metabolic terms from: {query}
        Return JSON format: {{"microbes": [], "metabolites": [], "health_terms": []}}
        """
        return self.llm.generate(prompt)
    
    def _run_cypher(self, query: str, params: dict) -> str:
        with self.driver.session() as session:
            result = session.run(query, params)
            return "\n".join([str(record) for record in result])