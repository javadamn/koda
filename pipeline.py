from crewai import Agent, Task, Crew
from llm_setup import get_llm 
from tools import execute_cypher_query_tool, get_graph_schema_tool 
from schema import GRAPH_SCHEMA_DESCRIPTION 
import config 

logger = config.get_logger(__name__)

class MicrobialAnalysisPipeline:
    """Encapsulates the CrewAI agents and tasks for microbial analysis."""
    def __init__(self):
        self.llm = get_llm() 

        # agents 
        self.query_constructor = Agent(
            role="Expert Neo4j Cypher Query Generator for Microbial Interactions",
            goal=f"""
                 Based on the user's question and the known graph schema, construct the most precise and efficient Cypher query(ies)
                 to retrieve the necessary data from the Neo4j knowledge graph.
                 Output MUST be a JSON string containing the 'query' and 'params' keys.
                 Example output format: '{{"query": "MATCH (m:Microbe {{name: $name}}) RETURN m.name, m.abundance", "params": {{"name": "Bacteroides_vulgatus"}}}}'
                 Use the provided schema: {GRAPH_SCHEMA_DESCRIPTION}
                 """,
            backstory="You are a bioinformatician specializing in graph databases. You have deep knowledge of the specific microbial interaction graph schema "
                      "and excel at translating natural language questions about microbes, metabolites, and pathways into effective Cypher queries.",
            tools=[get_graph_schema_tool], 
            llm=self.llm,
            verbose=True,
            memory=True,
            allow_delegation=False) #focus only on query generation

        self.information_retriever = Agent(
            role="Neo4j Database Query Executor",
            goal="Execute the provided Cypher query using the 'Execute Cypher Query Tool' and return the raw results or error message.",
            backstory="You are a database operator responsible for safely and efficiently executing queries against the Neo4j knowledge graph. "
                      "You only execute the queries given to you and pass back the results directly.",
            tools=[execute_cypher_query_tool],
            llm=self.llm,
            verbose=True,
            memory=False, #no need memory of past *results***, just the query to execute
            allow_delegation=False)

        self.contextual_analyzer = Agent(
            role="Microbial Ecology Data Analyst",
            goal=f"""
                 Analyze the data retrieved from the knowledge graph (provided in the context)
                 in light of the original user query (also provided). Synthesize the findings,
                 identify key patterns (e.g., important producers/consumers, high flux interactions, common pathways),
                 and explain the potential biological significance or implications.
                 If the data indicates an error or no results were found, state that clearly.
                 Use the schema context if needed: {GRAPH_SCHEMA_DESCRIPTION}
                 """,
            backstory="You are a microbial ecologist with expertise in interpreting metabolic network data. "
                      "You can take raw graph query results and explain what they mean in a biological context, "
                      "answering the user's specific questions and providing relevant insights.",
            tools=[], #anaalyzer interprets data, doesn't query again
            llm=self.llm,
            verbose=True,
            memory=True) #needs results back to the query

        self.report_writer = Agent(
            role="Scientific Report Writer",
            goal="Compile the analysis findings into a clear, concise, and well-structured report answering the original user query.",
            backstory="You are a scientific communicator skilled at summarizing complex analytical results "
                      "into an easily understandable report format, suitable for researchers or informed users.",
            tools=[],
            llm=self.llm,
            verbose=True,
            memory=False )

        self.agents = [
            self.query_constructor,
            self.information_retriever,
            self.contextual_analyzer,
            self.report_writer
                        ]

    def run_analysis(self, user_query: str) -> str:
        """Runs the microbial community analysis pipeline for a given user query."""

        logger.info(f"Starting analysis pipeline for query: '{user_query}'")

        #tasks
        construct_query_task = Task(
            description=f"""
                1. Analyze the user query: '{user_query}'
                2. Consult the graph schema (provided in your goal or use the schema tool if needed).
                3. Formulate the optimal Cypher query to answer the query.
                4. Output the query and any parameters as a JSON string.
                """,
            expected_output="A JSON string containing the 'query' and 'params' keys (e.g., '{\"query\": \"MATCH ...\", \"params\": {...}}').",
            agent=self.query_constructor
        )

        retrieve_data_task = Task(
            description="""
                1. Take the JSON string output from the previous task.
                2. Use the 'Execute Cypher Query Tool' to run the query against the database.
                3. Output the raw results (list of dictionaries) or the error dictionary returned by the tool.
                """,
            expected_output="A list of dictionaries representing the query results, or a dictionary containing an 'error' key.",
            agent=self.information_retriever,
            context=[construct_query_task] )

        analyze_results_task = Task(
            description=f"""
                1. Review the original user query: '{user_query}'
                2. Examine the data retrieved (or error message) from the previous task.
                3. If data exists, analyze it:
                    - Identify key microbes, metabolites, pathways mentioned or relevant.
                    - Describe the relationships found (production, consumption, cross-feeding, pathway involvement).
                    - Quantify findings using flux, abundance, or scores where available.
                    - Discuss potential biological implications or answer the specific question asked.
                4. If an error occurred or no data was found, clearly state this.
                5. Provide a detailed textual analysis.
                """,
            expected_output="A comprehensive textual analysis of the query results in the context of the user query, or a statement indicating missing data or errors.",
            agent=self.contextual_analyzer,
            context=[retrieve_data_task] )

        write_report_task = Task(
            description="""
                1. Take the textual analysis from the previous task.
                2. Synthesize the key findings into a concise and well-structured final report.
                3. Ensure the report directly addresses the original user query.
                4. Format the report clearly (e.g., using markdown).
                """,
            expected_output="A final, formatted report summarizing the analysis and answering the user query.",
            agent=self.report_writer,
            context=[analyze_results_task] )

        #run crew
        crew = Crew(
            agents=self.agents,
            tasks=[construct_query_task, retrieve_data_task, analyze_results_task, write_report_task],
            verbose=True 
            # sjould i  add process=Process.sequential? {if tasks must run strictly in order}
        )

        logger.info("Kicking off the Crew...")
        try:
            result = crew.kickoff()
            logger.info("Crew execution finished.")
           
            if hasattr(result, 'raw') and isinstance(result.raw, str):
                 return result.raw
            elif isinstance(result, str):
                 return result
            else:
                 logger.warning("Crew output format unexpected. Converting result object to string.")
                 return str(result) 
        except Exception as e:
            error_message = f"Critical error running the Crew: {e}"
            logger.exception(error_message)
            return error_message