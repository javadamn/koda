from crewai import Agent, Task, Process
from langchain.llms import Ollama  # or your preferred LLM
from neo4j_tools import Neo4jConnector

class MicrobialAgents:
    def __init__(self):
        self.neo4j = Neo4jConnector()  # Your Neo4j connection wrapper
        self.llm = Ollama(model="llama3")  # Consider bio-trained models like BioMistral
    
    def create_agents(self):
        return {
            "query_interpreter": Agent(
                role="Microbial Relationship Interpreter",
                goal="Translate user queries into graph-based biological relationships",
                backstory=("Specialized in interpreting biological queries and mapping them "
                          "to microbial cross-feeding relationships in the knowledge graph"),
                tools=[self.neo4j.query_graph, self.llm.tool],
                verbose=True
            ),
            "crossfeeding_analyst": Agent(
                role="Cross-Feeding Dynamics Specialist",
                goal="Analyze metabolite exchange patterns between microbial strains",
                backstory=("Expert in flux balance analysis interpretation and "
                          "microbial community metabolic interactions"),
                tools=[self.neo4j.advanced_cypher, self.llm.tool],
                verbose=True
            ),
            "hypothesis_generator": Agent(
                role="Microbial Interaction Hypothesizer",
                goal="Generate testable hypotheses about potential microbial interactions",
                backstory=("Creative thinker combining graph patterns with biological knowledge "
                          "to propose novel microbial relationships"),
                tools=[self.neo4j.pattern_finder, self.llm.tool],
                verbose=True
            ),
            "health_impact_analyst": Agent(
                role="Gut Health Impact Evaluator",
                goal="Connect microbial metabolites to host health outcomes",
                backstory=("Nutritional microbiologist expert in translating metabolic outputs "
                          "to physiological impacts"),
                tools=[self.neo4j.metabolite_tracer, self.llm.tool],
                verbose=True
            ),
            "report_synthesizer": Agent(
                role="Biological Insights Integrator",
                goal="Compile comprehensive reports from multiple analysis perspectives",
                backstory=("Systems biology expert skilled in synthesizing complex interactions "
                          "into actionable insights"),
                tools=[self.llm.tool],
                verbose=True
            )
        }