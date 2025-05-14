import json
import re
import ast
from typing import Dict, Any, List, Optional, Union
from crewai import Agent, Task, Crew, Process, TaskOutput
from llm_setup import get_llm
from tools import ExecuteCypherQueryToolClass, GetGraphSchemaToolClass
from schema import GRAPH_SCHEMA_DESCRIPTION
import config

logger = config.get_logger(__name__)

class MicrobialAnalysisPipeline:
    def __init__(self):
        self.llm = get_llm()
        self.execute_cypher_tool_instance = ExecuteCypherQueryToolClass()
        self.get_graph_schema_tool_instance = GetGraphSchemaToolClass() 
        self.crew_verbose_level_int = 1

        self.query_constructor = Agent(
            role="Expert Neo4j Cypher Query Generator for Microbial Interactions",
            goal=f"""
                 Based on the user's question and the known graph schema, construct the most precise and efficient Cypher query(ies)
                 to retrieve the necessary data from the Neo4j knowledge graph.
                 You MUST output ONLY a valid JSON string containing the 'query' and 'params' keys, without any preamble or conversational text.
                 Example JSON output format: '{{"query": "MATCH (m:microbe {{name: $name}}) RETURN m.name, m.abundance", "params": {{"name": "Bacteroides_vulgatus"}}}}'
                 Example query involving KOs: '{{"query": "MATCH (m:microbe)-[r:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(m.name)=toLower($m_name) RETURN k.name, r.description", "params": {{"m_name": "Bifidobacterium_longum"}}}}'
                 Use the provided schema: {GRAPH_SCHEMA_DESCRIPTION}
                 """,
            backstory="You are a bioinformatician specializing in graph databases. You have deep knowledge of the specific microbial interaction graph schema "
                      "and excel at translating natural language questions about microbes, metabolites, pathways, and KEGG Orthologies into effective Cypher queries.",
            tools=[self.get_graph_schema_tool_instance], 
            llm=self.llm,
            verbose=True,
            memory=True,
            allow_delegation=False
        )
        self.information_retriever = Agent(
            role="Neo4j Database Query Executor",
            goal="Execute the provided Cypher query using the 'Execute Cypher Query Tool' and return the raw results or error message. "
                 "Your final answer MUST be the direct output from the tool (a list of dictionaries representing data, or a dictionary with an 'error' key). "
                 "Do NOT add any conversational text, thoughts, or markdown formatting.",
            backstory="You are a database operator responsible for safely and efficiently executing queries against the Neo4j knowledge graph. "
                      "You only execute the queries given to you and pass back the results directly.", 
            tools=[self.execute_cypher_tool_instance],
            llm=self.llm,
            verbose=True,
            memory=False,
            allow_delegation=False
        )

        self.contextual_analyzer = Agent(
            role="Microbial Genomics and Drug Target Analyst",
            goal=f"""
                 Analyze the data retrieved from the knowledge graph (provided in the context)
                 in light of the original user query (also provided).
                 Specifically, if the query involves KEGG Orthologies (KOs) for a microbe:
                 1. List the identified KOs and their functional descriptions.
                 2. Emphasize that these KOs are associated with essential genes critical for the microbe's growth.
                 3. Discuss the implication of this essentiality: that these KOs represent potential drug targets for inhibiting the specific microbe.
                 4. If possible, categorize or highlight KOs that belong to well-known essential cellular processes (e.g., DNA replication, protein synthesis, cell wall synthesis, key metabolic pathways).
                 For other types of queries (metabolites, pathways), provide relevant biological context and insights.
                 Synthesize the findings, identify key patterns, and explain the potential biological significance.
                 If the data indicates an error or no results were found, state that clearly.
                 Your final output should be a detailed textual analysis.
                 Use the schema context if needed: {GRAPH_SCHEMA_DESCRIPTION}
                 """,
            backstory="You are an expert microbial and systems biologist with a focus on identifying novel antimicrobial drug targets. "
                      "You understand that KEGG Orthologies linked to essential genes in a microbe are prime candidates for such targets because their inhibition would likely impair microbial viability. "
                      "You can take raw graph query results listing KOs and their functions, and explain their significance in the context of essentiality and drug discovery, "
                      "answering the user's specific questions and providing relevant insights for further research.",
            tools=[],
            llm=self.llm,
            verbose=True,
            memory=True
        )

        self.report_writer = Agent(
            role="Scientific Report Writer",
            goal="Compile the analysis findings from the 'Microbial Genomics and Drug Target Analyst' into a clear, concise, and well-structured report answering the original user query. Your final output should be the formatted markdown report.",
            backstory="You are a scientific communicator skilled at summarizing complex analytical results, particularly those related to genomics and drug target identification, "
                      "into an easily understandable report format, suitable for researchers or informed users.",
            tools=[],
            llm=self.llm,
            verbose=True,
            memory=False
        )
        self.agents = [
            self.query_constructor,
            self.information_retriever,
            self.contextual_analyzer,
            self.report_writer
        ]

    def _get_agent_final_answer_from_task(self, task: Task) -> Optional[str]:
        """
        Safely extracts the agent's final answer string from a completed Task object.
        CrewAI stores the agent's final response in task.output.raw_output after kickoff.
        """
        if task.output is None:
            logger.warning(f"Task '{task.description[:50]}...' output is None.")
            return None
        
        if isinstance(task.output, TaskOutput):
            if hasattr(task.output, 'raw') and isinstance(task.output.raw, str):
                logger.info(f"Extracting from task.output.raw_output for task: {task.description[:50]}...")
                return task.output.raw.strip()
            if hasattr(task.output, 'expected_output') and task.output.expected_output:
                logger.info(f"Extracting from task.output.exported_output for task: {task.description[:50]}...")
                return str(task.output.expected_output).strip()
            if task.output.description and isinstance(task.output.description, str):
                 logger.info(f"Extracting from task.output.description for task: {task.description[:50]}...")
                 return task.output.description.strip()
            logger.warning(f"TaskOutput for '{task.description[:50]}...' did not have a usable raw_output, exported_output, or description string. Stringifying object.")
            return str(task.output).strip()
        elif isinstance(task.output, str):
            logger.info(f"Task output for '{task.description[:50]}...' is already a string.")
            return task.output.strip()
        else:
            logger.warning(f"Task output for '{task.description[:50]}...' is of unexpected type: {type(task.output)}. Stringifying.")
            return str(task.output).strip()

    def _extract_json_from_string(self, agent_final_answer: Optional[str]) -> Optional[str]:
        if not agent_final_answer:
            logger.warning("_extract_json_from_string: Received None or empty agent_final_answer.")
            return None
        
        logger.info(f"Attempting to extract JSON from agent_final_answer (first 500 chars): {agent_final_answer[:500]}...")
        
        match_md_json = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", agent_final_answer, re.DOTALL | re.IGNORECASE)
        if match_md_json:
            json_candidate = match_md_json.group(1)
            try:
                json.loads(json_candidate)
                logger.info(f"Extracted JSON from markdown block: {json_candidate[:200]}...")
                return json_candidate
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in markdown block: {json_candidate[:100]}")


        stripped_answer = agent_final_answer.strip()
        if (stripped_answer.startswith("'") and stripped_answer.endswith("'")) or \
           (stripped_answer.startswith('"') and stripped_answer.endswith('"')):
            potential_json_inner = stripped_answer[1:-1]
            if (potential_json_inner.startswith("{") and potential_json_inner.endswith("}")) or \
                (potential_json_inner.startswith("[") and potential_json_inner.endswith("]")):

                try:
                    json.loads(potential_json_inner) # Validate
                    logger.info(f"Extracted JSON after stripping outer quotes: {potential_json_inner[:200]}...")
                    return potential_json_inner
                except json.JSONDecodeError:
                    logger.warning(f"Stripping outer quotes did not result in valid JSON: {potential_json_inner[:100]}. Proceeding with original stripped_answer.")
            else: \
                pass
        
        if (stripped_answer.startswith("{") and stripped_answer.endswith("}")) or \
        (stripped_answer.startswith("[") and stripped_answer.endswith("]")): 
            try:
                json.loads(stripped_answer) #\
                logger.info(f"Extracted JSON directly from stripped agent answer: {stripped_answer[:200]}...")
                return stripped_answer
            except json.JSONDecodeError as e:
                logger.warning(f"Stripped agent answer looks like JSON but is invalid: {stripped_answer[:100]}. Error: {e}")

        logger.error(f"Could not extract clean JSON string from agent's final answer: {agent_final_answer[:300]}")
        return None

    def _extract_data_from_retriever_final_answer(self, retriever_agent_final_answer: Optional[str]) -> Union[List[Dict[str, Any]], Dict[str, str]]:
        if not retriever_agent_final_answer:
            return {"error": "InformationRetrieverAgent produced no final answer string."}

        logger.info(f"Attempting to parse data from retriever's final answer string: {retriever_agent_final_answer[:300]}...")
        stripped_content = retriever_agent_final_answer.strip()

        try:
            if (stripped_content.startswith('[') and stripped_content.endswith(']')) or \
            (stripped_content.startswith('{') and stripped_content.endswith('}')):
                try:
                    parsed_data = json.loads(stripped_content)
                    if isinstance(parsed_data, (list, dict)):
                        logger.info(f"Successfully parsed data from retriever's final answer using json.loads: Type {type(parsed_data)}")
                        return parsed_data
                except json.JSONDecodeError:
                    logger.warning(f"json.loads failed for retriever's output: '{stripped_content[:100]}...'. Attempting ast.literal_eval.")


            parsed_data = ast.literal_eval(stripped_content)
            if isinstance(parsed_data, (list, dict)):
                logger.info(f"Successfully parsed data from retriever's final answer using ast.literal_eval: Type {type(parsed_data)}")
                return parsed_data
            else:
                logger.warning(f"ast.literal_eval resulted in non-list/dict type: {type(parsed_data)} for content: {stripped_content[:200]}")
                return {"error": f"Retriever's final answer (via ast.literal_eval) not list/dict. Type: {type(parsed_data)}", "raw": stripped_content}

        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as e:
            logger.error(f"Error parsing retriever's final answer (json.loads or ast.literal_eval): {e}. Content: {stripped_content}", exc_info=True)
            return {"error": f"Failed to parse retriever's final answer: {e}", "raw": stripped_content}
        except Exception as e: 
            logger.error(f"Unexpected error parsing retriever's final answer: {e}. Content: {stripped_content}", exc_info=True)
            return {"error": f"Unexpected error processing retriever's final answer: {e}", "raw": stripped_content}

    def run_analysis(self, user_query: str) -> Dict[str, Any]:
        logger.info(f"Main Pipeline: Starting analysis for query: '{user_query}'")
        
        construct_query_task = Task(
            description=f"Analyze the user query: '{user_query}'. Consult schema (use 'Get Graph Schema Tool' if needed - it takes no specific input from you). Formulate Cypher. Your final answer MUST be ONLY the JSON string with 'query' and 'params'.",
            expected_output="A valid JSON string with 'query' and 'params' keys ONLY.",
            agent=self.query_constructor,
        )
        retrieve_data_task = Task(
            description=f"Take the JSON string from 'construct_query_task'. Parse it. Use 'Execute Cypher Query Tool'. Your final answer MUST be ONLY the tool's direct output (list of dicts or error dict).",
            expected_output="A list of dictionaries (data) or an error dictionary.",
            agent=self.information_retriever, context=[construct_query_task]
        )
        analyze_results_task = Task(
            description=f"Review NLQ: '{user_query}'. Examine data from 'retrieve_data_task'. Analyze as 'Microbial Genomics and Drug Target Analyst'. Your final answer should be ONLY the detailed textual analysis.",
            expected_output="Comprehensive textual analysis or error statement.",
            agent=self.contextual_analyzer, context=[retrieve_data_task]
        )
        write_report_task = Task(
            description="Take analysis from 'analyze_results_task'. Synthesize into a report. Your final answer should be ONLY the markdown report.",
            expected_output="Final markdown formatted report.",
            agent=self.report_writer, context=[analyze_results_task]
        )

        crew_verbose_bool = True if self.crew_verbose_level_int > 0 else False
        crew = Crew(
            agents=self.agents,
            tasks=[construct_query_task, retrieve_data_task, analyze_results_task, write_report_task],
            process=Process.sequential, verbose=crew_verbose_bool
        )

        logger.info("Main Pipeline: Kicking off the Crew...")
        crew.kickoff() 
        logger.info("Main Pipeline: Crew execution finished.")

        qc_agent_final_answer_str = self._get_agent_final_answer_from_task(construct_query_task)
        generated_cypher_json_str_output = self._extract_json_from_string(qc_agent_final_answer_str)
        if not generated_cypher_json_str_output:
            generated_cypher_json_str_output = json.dumps({
                "error": "Failed to extract valid Cypher JSON from QueryConstructorAgent",
                "raw_agent_final_answer": qc_agent_final_answer_str or "No output string from QueryConstructorAgent task."
            })

        retriever_agent_final_answer_str = self._get_agent_final_answer_from_task(retrieve_data_task)
        retrieved_data_actual = self._extract_data_from_retriever_final_answer(retriever_agent_final_answer_str)
        
        final_report_output_str = self._get_agent_final_answer_from_task(write_report_task)
        if not final_report_output_str:
            final_report_output_str = "Error: Could not retrieve final report string from write_report_task."

        logger.info(f"Main Pipeline: Extracted Cypher JSON: {generated_cypher_json_str_output}")
        logger.info(f"Main Pipeline: Extracted Retrieved Data (type {type(retrieved_data_actual)}): {str(retrieved_data_actual)[:200]}...")
        logger.info(f"Main Pipeline: Extracted Final Report (first 200 chars): {final_report_output_str[:200]}...")

        return {
            "generated_cypher_json_str": generated_cypher_json_str_output,
            "retrieved_data": retrieved_data_actual,
            "final_report": final_report_output_str
        }