
GRAPH_SCHEMA_DESCRIPTION = """
The knowledge graph contains information about microbial interactions.

Key Node Labels:
- Microbe: Represents a microbial strain.
  - Properties:
    - name (string, unique identifier)
    - abundance (float, optional, relative presence of the microbe)
- Metabolite: Represents a chemical compound involved in interactions.
  - Properties:
    - name (string, unique identifier)
- Pathway: Represents a biological pathway microbes are involved in.
  - Properties:
    - name (string, unique identifier)

Key Relationship Types (with properties):
- [:PRODUCES] - (Microbe)-[:PRODUCES {flux: float, description: str}]->(Metabolite)
- [:CONSUMES] - (Metabolite)-[:CONSUMES {flux: float, description: str}]->(Microbe) # Note: Relationship direction in graph storage. Query typically reverses: (Microbe)<-[:CONSUMES]-(Metabolite)
- [:CROSS_FEEDS_WITH] - (Microbe)-[:CROSS_FEEDS_WITH {source_biomass: float, target_biomass: float}]->(Microbe)
- [:INVOLVED_IN] - (Microbe)-[:INVOLVED_IN {subsystem_score: float, description: str}]->(Pathway)

Important Considerations for Queries:
- Always use parameters ($param_name) for node names or other values in WHERE clauses for security and efficiency.
- **Case Sensitivity**: Node name properties (e.g., Metabolite.name, Microbe.name) might have inconsistent capitalization. To ensure reliable matching regardless of case, **always use the `toLower()` function** on both the property and the parameter when matching names. Example: `WHERE toLower(met.name) = toLower($name)`.
- When searching for microbes related to a metabolite, consider both PRODUCES and CONSUMES relationships.
- Flux values indicate the rate of production/consumption. Higher flux might indicate higher importance.
- Abundance represents the relative presence of a microbe (if available).
- Subsystem score represents the importance of a pathway to a microbe.

Specific Query Patterns (Using Case-Insensitive Matching):
- To find entities doing *both* A and B (e.g., produce AND consume a specific metabolite):
  MATCH (met:Metabolite) WHERE toLower(met.name) = toLower($name) // Case-insensitive match first
  WITH met // Pass the matched metabolite
  MATCH (m:Microbe)-[:PRODUCES]->(met)
  MATCH (m)<-[:CONSUMES]-(met) // Find microbes with both relationships to 'met'
  RETURN m.name

- To calculate net values (e.g., net flux = production - consumption) for microbes doing *both*:
  MATCH (met:Metabolite) WHERE toLower(met.name) = toLower($name) // Case-insensitive match first
  WITH met
  MATCH (m:Microbe)-[p:PRODUCES]->(met)
  MATCH (m)<-[c:CONSUMES]-(met)
  RETURN m.name, p.flux AS production_flux, c.flux AS consumption_flux, (p.flux - c.flux) AS net_flux

- To handle cases where production or consumption might be missing (find microbes doing *either or both*), use OPTIONAL MATCH and COALESCE:
  MATCH (met:Metabolite) WHERE toLower(met.name) = toLower($name) // Case-insensitive match first
  WITH met
  MATCH (m:Microbe) // Consider adding WHERE clause if microbe list is too large
  OPTIONAL MATCH (m)-[p:PRODUCES]->(met)
  OPTIONAL MATCH (m)<-[c:CONSUMES]-(met)
  // Filter for microbes that have at least one interaction with the metabolite
  WHERE p IS NOT NULL OR c IS NOT NULL
  WITH m, met, COALESCE(p.flux, 0.0) AS production_flux, COALESCE(c.flux, 0.0) AS consumption_flux
  RETURN m.name, production_flux, consumption_flux, (production_flux - consumption_flux) AS net_flux
  ORDER BY net_flux DESC // Example ordering

Ensure all variables used in RETURN or calculations are defined in the preceding MATCH or WITH clauses. Use WITH clauses effectively to pass variables between MATCH clauses.
"""