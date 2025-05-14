import json
import time
import re
from typing import Dict, Any, List, Optional, Union

from pydantic import BaseModel, Field, AliasChoices, ConfigDict, ValidationError
from crewai import Agent, Task, Crew, Process, TaskOutput 
from langchain_openai import ChatOpenAI

from pipeline1 import MicrobialAnalysisPipeline 
from schema import GRAPH_SCHEMA_DESCRIPTION
import config

EVAL_LLM_MODEL = config.LLM_MODEL
EVAL_LLM_TEMPERATURE = 0.1
logger = config.get_logger("EvaluationPipeline")

# --- ##::Pydantic Models ---
class CypherReviewSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='ignore')
    syntactic_validity: int = Field(..., ge=1, le=5, validation_alias=AliasChoices('syntactic_validity', 'Syntactic Validity'))
    schema_adherence: int = Field(..., ge=1, le=5, validation_alias=AliasChoices('schema_adherence', 'Schema Adherence'))
    semantic_accuracy_nlq: int = Field(..., ge=1, le=5, validation_alias=AliasChoices('semantic_accuracy_nlq', 'Semantic Accuracy vs. NLQ'))
    semantic_accuracy_gold: Optional[int] = Field(default=None, ge=1, le=5, validation_alias=AliasChoices('semantic_accuracy_gold', 'Semantic Accuracy vs. Gold Standard Query'))
    parameterization: int = Field(..., ge=1, le=5, validation_alias=AliasChoices('parameterization', 'Correct Parameterization'))
    tolower_usage: int = Field(..., ge=1, le=5, validation_alias=AliasChoices('tolower_usage', 'Use of toLower() for Names', 'Use of `toLower()` for Names'))
    qualitative_feedback: str = Field(..., validation_alias=AliasChoices('qualitative_feedback', 'Comments', 'qualitative_feedback'))
    is_executable_in_neo4j: Optional[bool] = Field(default=None, validation_alias=AliasChoices('is_executable_in_neo4j', 'Is Executable in Neo4j'))

class AnalysisReviewSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='ignore')
    factual_accuracy_grounding: int = Field(..., ge=1, le=5, validation_alias=AliasChoices('factual_accuracy_grounding', 'Factual Accuracy & Grounding', "FactualAccuracyAndGrounding"))
    relevance_completeness_nlq: int = Field(..., ge=1, le=5, validation_alias=AliasChoices('relevance_completeness_nlq', 'Relevance & Completeness for NLQ', "RelevanceAndCompletenessForNLQ"))
    depth_insight_scientific_value: int = Field(..., ge=1, le=5, validation_alias=AliasChoices('depth_insight_scientific_value', 'Depth of Insight & Scientific Value', "DepthOfInsightAndScientificValue"))
    clarity_coherence_structure: int = Field(..., ge=1, le=5, validation_alias=AliasChoices('clarity_coherence_structure', 'Clarity, Coherence, & Structure', "ClarityCoherenceAndStructure"))
    drug_target_discussion_quality: Optional[int] = Field(default=None, ge=1, le=5, validation_alias=AliasChoices('drug_target_discussion_quality', 'Drug Target Discussion Quality'))
    qualitative_feedback: str = Field(..., validation_alias=AliasChoices('qualitative_feedback', 'Comments'))

def get_eval_llm():
    return ChatOpenAI(model_name=EVAL_LLM_MODEL, temperature=EVAL_LLM_TEMPERATURE, openai_api_key=config.OPENAI_API_KEY)

# --- ##::Reviewer Agents ---
cypher_query_reviewer_agent = Agent(
    role="Expert Cypher Query Reviewer and Neo4j Specialist",
    goal=f"""
        Critically evaluate a given Cypher query based on multiple criteria.
        You MUST provide your evaluation as a valid JSON object that strictly conforms to the CypherReviewSchema.
        Ensure all JSON keys are exactly as defined in the schema (e.g., 'syntactic_validity', 'schema_adherence'). Do NOT use spaces or PascalCase for keys.
        The following fields MUST be integers on a scale of 1 (poor) to 5 (excellent):
    - 'syntactic_validity'
    - 'schema_adherence'
    - 'semantic_accuracy_nlq'
    - 'semantic_accuracy_gold' (if applicable, otherwise null or omit)
    - 'parameterization'
    - 'tolower_usage'
    The 'qualitative_feedback' field MUST be a string containing your detailed comments.
    The 'is_executable_in_neo4j' field MUST be a boolean (true/false) or null/omitted if unknown.
    Your entire response should be ONLY the JSON object, without any surrounding text, thoughts, or markdown backticks.
    GRAPH SCHEMA for reference:
    ---
    {GRAPH_SCHEMA_DESCRIPTION}
    ---
    """,
    backstory="You are a meticulous Neo4j expert...",
    llm=get_eval_llm(), verbose=True, allow_delegation=False,
    output_json_parser=CypherReviewSchema # CrewAI will try to parse output into this model
)

analysis_report_reviewer_agent = Agent(
    role="Scientific Report Reviewer and Microbial Genomics Expert",
    goal=f"""
        Critically evaluate a textual analysis report.
    You MUST provide your evaluation as a valid JSON object that strictly conforms to the AnalysisReviewSchema.
    Ensure all JSON keys are exactly as defined in the schema.
    The following fields MUST be integers on a scale of 1 (poor) to 5 (excellent):
    - 'factual_accuracy_grounding'
    - 'relevance_completeness_nlq'
    - 'depth_insight_scientific_value'
    - 'clarity_coherence_structure'
    - 'drug_target_discussion_quality' (if applicable, otherwise null or omit)
    The 'qualitative_feedback' field MUST be a string containing your detailed comments.
    Your entire response should be ONLY the JSON object, without any surrounding text, thoughts, or markdown backticks.
    """,
    backstory="You are a seasoned researcher in microbial genomics...",
    llm=get_eval_llm(), verbose=True, allow_delegation=False,
    output_json_parser=AnalysisReviewSchema 
)

def _get_agent_final_answer_str_from_task_output_eval(task_output_obj: Any) -> Optional[str]:
    """
    Safely extracts the agent's 'Final Answer' string from a TaskOutput object for evaluation agents.
    Reviewer agents have output_json_parser, so their output might be directly the model or dict.
    """
    if task_output_obj is None: return None
    
    if isinstance(task_output_obj, BaseModel):
        try: return task_output_obj.model_dump_json()
        except Exception: return str(task_output_obj)
    if isinstance(task_output_obj, dict):
        try: return json.dumps(task_output_obj)
        except Exception: return str(task_output_obj)
    if isinstance(task_output_obj, str): return task_output_obj

    text_to_search = ""
    if isinstance(task_output_obj, TaskOutput):
        if task_output_obj.expected_output:
            if isinstance(task_output_obj.expected_output, str): return task_output_obj.expected_output
            if isinstance(task_output_obj.expected_output, dict): return json.dumps(task_output_obj.expected_output)
            if isinstance(task_output_obj.expected_output, BaseModel): return task_output_obj.expected_output.model_dump_json()
        if task_output_obj.description and isinstance(task_output_obj.description, str):
            text_to_search = task_output_obj.description
        elif hasattr(task_output_obj, 'raw') and isinstance(task_output_obj.raw, str):
            text_to_search = task_output_obj.raw
        else: text_to_search = str(task_output_obj)
    else: text_to_search = str(task_output_obj)

    logger.debug(f"_get_agent_final_answer_str_from_task_output_eval: Text to search (first 500 chars): {text_to_search[:500]}")
    json_block_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text_to_search, re.IGNORECASE | re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()
    
    return text_to_search.strip()


def _extract_and_validate_json_review(task_output_obj: Any, pydantic_model: type[BaseModel]) -> Dict[str, Any]:
    """Extracts JSON review from a reviewer agent's task output and validates with Pydantic."""
    
    if isinstance(task_output_obj, pydantic_model):
        logger.info(f"Review output is already a validated Pydantic model: {pydantic_model.__name__}")
        return task_output_obj.model_dump()
    
    if isinstance(task_output_obj, dict):
        logger.info(f"Review output is dict, attempting to validate directly with {pydantic_model.__name__}")
        try:
            common_schema_key_present = any(
                alias in task_output_obj 
                for field_info in pydantic_model.model_fields.values()
                for alias in ([field_info.alias] if field_info.alias else []) + ([field_info.name] if field_info.name else [])
            )
            is_task_output_structure = 'raw' in task_output_obj and 'agent' in task_output_obj

            if common_schema_key_present and not is_task_output_structure:
                model_instance = pydantic_model.model_validate(task_output_obj)
                logger.info(f"Successfully validated dict output directly against {pydantic_model.__name__}")
                return model_instance.model_dump()
            else:
                logger.info(f"Dict output does not appear to be the direct model or is TaskOutput structure. Will process via _get_agent_final_answer_str_from_task_output_eval.")
        except ValidationError: #if direct validation fails >>>> proceed to parse it as a string.
            logger.warning(f"Direct Pydantic validation of dict failed for {pydantic_model.__name__}. Will process via _get_agent_final_answer_str_from_task_output_eval. Dict: {str(task_output_obj)[:300]}")


    agent_final_answer_str = _get_agent_final_answer_str_from_task_output_eval(task_output_obj)

    if not agent_final_answer_str:
        logger.error(f"Could not extract a string from task_output_obj for {pydantic_model.__name__}.")
        return {"error": f"No parsable string output from task for {pydantic_model.__name__}.", "raw_task_output": str(task_output_obj)}

    logger.info(f"Attempting to parse and validate review output for {pydantic_model.__name__} (from agent's final answer string): {agent_final_answer_str[:500]}...")
    
    try:
        #::parse the string representation of the TaskOutput object (or the direct JSON)
        parsed_outer_dict = json.loads(agent_final_answer_str)

        actual_review_dict_to_validate = None

        if isinstance(parsed_outer_dict, dict):
            if 'raw' in parsed_outer_dict and isinstance(parsed_outer_dict['raw'], str) \
               and 'agent' in parsed_outer_dict: 
                
                json_str_from_raw_key = parsed_outer_dict['raw']
                logger.info(f"Detected TaskOutput-like structure. Extracting review JSON from 'raw' key for {pydantic_model.__name__}: {json_str_from_raw_key[:300]}...")
                try:
                    actual_review_dict_to_validate = json.loads(json_str_from_raw_key)
                except json.JSONDecodeError as e_inner:
                    logger.error(f"JSONDecodeError parsing content of 'raw' key for {pydantic_model.__name__}: {e_inner}. String from 'raw': {json_str_from_raw_key}")
                    return {
                        "error": f"Failed to parse JSON from 'raw' key for {pydantic_model.__name__}.",
                        "raw_key_content": json_str_from_raw_key,
                        "original_agent_output_string": agent_final_answer_str
                    }
            else:
                logger.info(f"Parsed agent's final answer string directly into a dictionary for {pydantic_model.__name__}. Assuming this is the review content.")
                actual_review_dict_to_validate = parsed_outer_dict
        else:
            logger.error(f"Parsing agent_final_answer_str did not yield a dictionary for {pydantic_model.__name__}. Type: {type(parsed_outer_dict)}. String: {agent_final_answer_str[:300]}")
            return {"error": f"Parsed agent_final_answer_str is not a dictionary for {pydantic_model.__name__}.", "raw": agent_final_answer_str}

        if actual_review_dict_to_validate is not None:
            model_instance = pydantic_model.model_validate(actual_review_dict_to_validate)
            logger.info(f"Successfully validated JSON against {pydantic_model.__name__}.")
            return model_instance.model_dump()
        else: #should not happen if logic is correct, but as a safeguard
            logger.error(f"Could not determine the actual review dictionary to validate for {pydantic_model.__name__}.")
            return {"error": f"Could not isolate actual review dictionary for {pydantic_model.__name__}.", "raw": agent_final_answer_str}

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError parsing agent_final_answer_str for {pydantic_model.__name__}: {e}. String used: {agent_final_answer_str[:500]}")
        return {"error": f"Failed to parse agent_final_answer_str as JSON for {pydantic_model.__name__}.", "raw": agent_final_answer_str}
    except ValidationError as ve:
        dict_being_validated_str = str(actual_review_dict_to_validate if 'actual_review_dict_to_validate' in locals() and actual_review_dict_to_validate is not None else agent_final_answer_str)
        logger.error(f"Pydantic ValidationError for {pydantic_model.__name__}: {ve}. Content attempted for validation: {dict_being_validated_str[:500]}")
        return {
            "error": f"Pydantic validation failed for {pydantic_model.__name__}.",
            "raw_content_validated": dict_being_validated_str,
            "validation_error_details": str(ve),
            "original_agent_output_string": agent_final_answer_str if 'actual_review_dict_to_validate' in locals() else None #only add original if different
        }
    except Exception as e_unexp:
        logger.error(f"Unexpected error in _extract_and_validate_json_review for {pydantic_model.__name__}: {e_unexp}. String used: {agent_final_answer_str[:500]}", exc_info=True)
        return {"error": f"Unexpected error during review validation for {pydantic_model.__name__}.", "raw": agent_final_answer_str}

def evaluate_pipeline_output(
    nlq: str,
    generated_query_json_str: str,
    retrieved_data: Union[List[Dict[str, Any]], Dict[str, Any]],
    generated_report: str,
    gold_standard_cypher: Optional[str] = None,
) -> Dict[str, Any]:
    evaluation_results = {}
    parsed_query_dict_for_reviewer = {} 

    if not generated_query_json_str or not isinstance(generated_query_json_str, str):
        logger.error(f"Invalid generated_query_json_str for NLQ '{nlq}': Type {type(generated_query_json_str)}, Value: {generated_query_json_str}")
        evaluation_results["query_review"] = {"error": "Invalid or missing generated Cypher JSON string from main pipeline."}
    else:
        try:
            parsed_query_dict_for_reviewer = json.loads(generated_query_json_str)
            if "error" in parsed_query_dict_for_reviewer: 
                 logger.error(f"The 'generated_query_json_str' from main pipeline contained an error: {parsed_query_dict_for_reviewer['error']}")
                 evaluation_results["query_review"] = parsed_query_dict_for_reviewer 
        except json.JSONDecodeError:
            logger.error(f"Could not parse generated_query_json_str for NLQ '{nlq}': {generated_query_json_str}")
            evaluation_results["query_review"] = {"error": "Generated Cypher by main pipeline is not valid JSON.", "raw_query_str": generated_query_json_str}

    query_review_task = None
    if "error" not in evaluation_results.get("query_review", {}):
        query_review_task_description = f"""
            Natural Language Question (NLQ):
            "{nlq}"

            Generated Cypher Query to review:
            Query: "{parsed_query_dict_for_reviewer.get('query', 'QUERY STRING NOT FOUND IN PARSED JSON')}"
            Params: {json.dumps(parsed_query_dict_for_reviewer.get('params', {}))}
        """
        if gold_standard_cypher:
            query_review_task_description += f"""
            Gold Standard Cypher Query (for semantic comparison):
            {gold_standard_cypher}
            """
        query_review_task_description += """
            Based on the NLQ, the generated query, the (optional) gold standard query, and the GRAPH SCHEMA provided in your role,
            evaluate the generated Cypher query according to all criteria mentioned in your goal.
            Your output MUST be a valid JSON object conforming to the CypherReviewSchema, using snake_case for all keys.
        """
        query_review_task = Task(
            description=query_review_task_description,
            expected_output="A JSON object conforming to CypherReviewSchema.",
            agent=cypher_query_reviewer_agent,
        )
    else:
        logger.info(f"Skipping Cypher query review for NLQ '{nlq}' due to prior error in query generation/parsing from main pipeline.")


    retrieved_data_for_llm_str = ""
    if isinstance(retrieved_data, dict) and "error" in retrieved_data:
        retrieved_data_for_llm_str = f"Data Retrieval Error from Main Pipeline: {retrieved_data['error']}"
        if "raw" in retrieved_data: 
             retrieved_data_for_llm_str += f" Raw content from main pipeline: {str(retrieved_data['raw'])[:200]}"
    elif isinstance(retrieved_data, list):
        retrieved_data_for_llm_str = json.dumps(retrieved_data[:10], indent=2)
        if len(retrieved_data) > 10:
            retrieved_data_for_llm_str += f"\n... and {len(retrieved_data) - 10} more records."
    elif retrieved_data is None:
        retrieved_data_for_llm_str = "No data was retrieved by the main pipeline (retrieved_data is None)."
    else:
        retrieved_data_for_llm_str = f"Retrieved data from main pipeline is in an unexpected format: {type(retrieved_data)}"
        logger.warning(f"Retrieved data for LLM reviewer is in unexpected format: {type(retrieved_data)}. Content: {str(retrieved_data)[:200]}")

    analysis_review_task_description = f"""
        Natural Language Question (NLQ):
        "{nlq}"

        Retrieved Data (from executing the generated Cypher query by the main pipeline):
        {retrieved_data_for_llm_str}

        Generated Analysis Report (by the main pipeline):
        ---
        {generated_report}
        ---

        Based on the NLQ, the retrieved data, and the generated report,
        evaluate the report according to all criteria mentioned in your goal, paying special attention to the discussion of KOs as essential genes and drug targets if applicable.
        Your output MUST be a valid JSON object conforming to the AnalysisReviewSchema, using snake_case for all keys.
    """
    analysis_review_task = Task(
        description=analysis_review_task_description,
        expected_output="A JSON object conforming to AnalysisReviewSchema.",
        agent=analysis_report_reviewer_agent,
    )

    tasks_to_run = []
    if query_review_task: tasks_to_run.append(query_review_task)
    tasks_to_run.append(analysis_review_task) 
    
    if not tasks_to_run: 
        logger.error("No evaluation tasks to run.")
        return {"query_review": evaluation_results.get("query_review"), "analysis_review": {"error": "No analysis review task was run."}}

    evaluation_crew = Crew(
        agents=[cypher_query_reviewer_agent, analysis_report_reviewer_agent],
        tasks=tasks_to_run,
        process=Process.sequential,
        verbose=1
    )

    logger.info(f"Starting evaluation tasks for NLQ: {nlq}")
    crew_kickoff_result = evaluation_crew.kickoff() 

    evaluation_results["query_review"] = _extract_and_validate_json_review(query_review_task.output if query_review_task else None, CypherReviewSchema) \
                                         if query_review_task else evaluation_results.get("query_review", {"error": "Query review skipped due to earlier error."})
    evaluation_results["analysis_review"] = _extract_and_validate_json_review(analysis_review_task.output, AnalysisReviewSchema)
    
    return evaluation_results

if __name__ == "__main__":
    try:
        main_pipeline = MicrobialAnalysisPipeline()
        logger.info("Main MicrobialAnalysisPipeline initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize MicrobialAnalysisPipeline: {e}", exc_info=True)
        exit()

    benchmark_items = [
        {
            "nlq": "What KEGG Orthologies (KOs) are associated with the microbe Klebsiella_pneumoniae_pneumoniae_MGH78578 and what are their functional descriptions?",
            "gold_standard_cypher": "MATCH (m:microbe)-[r:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(m.name) = toLower('Klebsiella_pneumoniae_pneumoniae_MGH78578') RETURN k.name AS ko_id, r.description AS ko_functional_description"
        },
        {
            "nlq": "Which microbes produce Thiamine and also have KOs whose description mentions 'synthase'?",
            "gold_standard_cypher": "MATCH (m:microbe)-[:PRODUCES]->(met:metabolite) WHERE toLower(met.name) = toLower('Thiamine') WITH m MATCH (m)-[r_ko:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(r_ko.description) CONTAINS toLower('synthase') RETURN DISTINCT m.name AS microbe_name"
        },
        {
            "nlq": "How many distinct KOs are associated with Klebsiella_pneumoniae_pneumoniae_MGH78578?",
            "gold_standard_cypher": "MATCH (m:microbe)-[:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(m.name) = toLower('Klebsiella_pneumoniae_pneumoniae_MGH78578') RETURN count(DISTINCT k.name) AS distinct_ko_count"
        },    
        {
            "nlq": "What KOs are found in microbes that consume cetic acid, and what are the descriptions of these KO relationships?",
            "gold_standard_cypher": "MATCH (m:microbe)<-[:CONSUMES]-(met:metabolite) WHERE toLower(met.name) = toLower('cetic acid') WITH m MATCH (m)-[r_ko:HAS_KEGG_ORTHOLOGY]->(k:KO) RETURN DISTINCT m.name AS microbe_name, k.name AS ko_id, r_ko.description AS ko_functional_description"
        },
        {
            "nlq": "Identify microbes that Bifidobacterium_adolescentis_ATCC_15703 cross-feeds with (where it is the source), and list any KOs these target microbes have related to 'NAD Synthase'.",
            "gold_standard_cypher": "MATCH (source_microbe:microbe)-[:CROSS_FEEDS_WITH]->(target_microbe:microbe) WHERE toLower(source_microbe.name) = toLower('Bifidobacterium_adolescentis_ATCC_15703') WITH target_microbe MATCH (target_microbe)-[r_ko:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(r_ko.description) CONTAINS toLower('NAD Synthase') RETURN DISTINCT target_microbe.name AS target_microbe, k.name AS ko_id, r_ko.description AS ko_functional_description"
        },
        {
            "nlq": "List all KOs for Bacteroides_fragilis_ATCC_25285 and all KOs for Parabacteroides_distasonis_ATCC_8503.",
            "gold_standard_cypher": "MATCH (m:microbe)-[r:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(m.name) IN [toLower('Bacteroides_fragilis_ATCC_25285'), toLower('Parabacteroides_distasonis_ATCC_8503')] RETURN m.name AS microbe_name, k.name AS ko_id, r.description AS ko_functional_description"
        },
        {
            "nlq": "What metabolites are produced by microbes that do not possess the KO K00130 (pyruvate kinase)?",
            "gold_standard_cypher": "MATCH (m:microbe) WHERE NOT (m)-[:HAS_KEGG_ORTHOLOGY]->(:KO {name: 'K00130'}) WITH m MATCH (m)-[:PRODUCES]->(met:metabolite) RETURN DISTINCT m.name AS microbe_name, met.name AS produced_metabolite ORDER BY microbe_name, produced_metabolite"
        },
        {
            "nlq": "Show me KOs related to 'NAD Synthase' that are found in microbes and list the microbe names.",
            "gold_standard_cypher": "MATCH (m:microbe)-[r:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(r.description) CONTAINS toLower('NAD Synthase') RETURN DISTINCT k.name AS ko_id, r.description AS ko_functional_description, m.name AS microbe_name, m.abundance ORDER BY m.abundance ASC"
        },
        {
            "nlq": "Which microbes consume 'Acetic acid' and are involved in the 'Fatty acid synthesis' with a score above 50?",
            "gold_standard_cypher": "MATCH (m:microbe)<-[:CONSUMES]-(met:metabolite) WHERE toLower(met.name) = toLower('Acetic acid') WITH m MATCH (m)-[inv:INVOLVED_IN]->(p:pathway) WHERE toLower(p.name) = toLower('Fatty acid synthesis') AND inv.subsystem_score > 50 RETURN DISTINCT m.name AS microbe_name, inv.subsystem_score AS subsystem_score"
        }
]


    all_evaluations = []
    evaluation_comments_text = "" 

    for item_idx, item in enumerate(benchmark_items):
        nlq = item["nlq"]
        logger.info(f"\n--- Evaluating NLQ {item_idx + 1}/{len(benchmark_items)}: {nlq} ---")
        evaluation_comments_text += f"\n\n--- NLQ: {nlq} ---\n"

        pipeline_outputs = main_pipeline.run_analysis(nlq)

        generated_cypher_json_str = pipeline_outputs.get("generated_cypher_json_str")
        retrieved_data = pipeline_outputs.get("retrieved_data")
        final_report = pipeline_outputs.get("final_report")

        if not isinstance(pipeline_outputs, dict) or \
           not all(k in pipeline_outputs for k in ["generated_cypher_json_str", "retrieved_data", "final_report"]):
            logger.error(f"Main pipeline did not return the expected dictionary structure for NLQ: {nlq}. Outputs: {str(pipeline_outputs)[:500]}")
            error_info = {"nlq": nlq, "error": "Incomplete or malformed outputs from main pipeline."}
            all_evaluations.append(error_info)
            evaluation_comments_text += f"ERROR: Incomplete outputs from main pipeline.\nPipeline Outputs: {str(pipeline_outputs)[:500]}\n"
            continue
        
        logger.info(f"Main pipeline generated Cypher JSON string: {generated_cypher_json_str}")
        if isinstance(retrieved_data, list):
            logger.info(f"Main pipeline retrieved data (first 2 records if any): {retrieved_data[:2]}")
        else:
            logger.info(f"Main pipeline retrieved data (or error): {str(retrieved_data)[:200]}")
        logger.info(f"Main pipeline generated report (first 100 chars): {final_report[:100]}...")

        evaluation = evaluate_pipeline_output(
            nlq=nlq,
            generated_query_json_str=str(generated_cypher_json_str),
            retrieved_data=retrieved_data,
            generated_report=str(final_report),
            gold_standard_cypher=item.get("gold_standard_cypher")
        )
        
        serializable_pipeline_outputs = {
            "generated_cypher_json_str": str(generated_cypher_json_str),
            "retrieved_data": retrieved_data if isinstance(retrieved_data, (list, dict)) else str(retrieved_data),
            "final_report": str(final_report)
        }
        all_evaluations.append({"nlq": nlq, "evaluation": evaluation, "pipeline_outputs_from_main": serializable_pipeline_outputs})
        logger.info(f"Evaluation for '{nlq}': {json.dumps(evaluation, indent=2)}")

        query_review_data = evaluation.get("query_review", {})
        if isinstance(query_review_data, dict):
            if "error" not in query_review_data:
                evaluation_comments_text += f"\nQuery Review Feedback:\n{query_review_data.get('qualitative_feedback', 'N/A')}\n"
                scores = {k:v for k,v in query_review_data.items() if k != 'qualitative_feedback'}
                if scores: evaluation_comments_text += f"Scores: {scores}\n"
            else:
                evaluation_comments_text += f"\nQuery Review Error: {query_review_data['error']}\nRaw: {query_review_data.get('raw', query_review_data.get('raw_query_str', query_review_data.get('raw_agent_answer', query_review_data.get('raw_parsed_content', 'N/A'))))}\n"
        else:
            evaluation_comments_text += f"\nQuery Review output was not a dictionary: {query_review_data}\n"

        analysis_review_data = evaluation.get("analysis_review", {})
        if isinstance(analysis_review_data, dict):
            if "error" not in analysis_review_data:
                evaluation_comments_text += f"\nAnalysis Review Feedback:\n{analysis_review_data.get('qualitative_feedback', 'N/A')}\n"
                scores = {k:v for k,v in analysis_review_data.items() if k != 'qualitative_feedback'}
                if scores: evaluation_comments_text += f"Scores: {scores}\n"
            else:
                evaluation_comments_text += f"\nAnalysis Review Error: {analysis_review_data['error']}\nRaw: {analysis_review_data.get('raw', analysis_review_data.get('raw_parsed_content', 'N/A'))}\n"
        else:
            evaluation_comments_text += f"\nAnalysis Review output was not a dictionary: {analysis_review_data}\n"

        if item_idx < len(benchmark_items) - 1:
             logger.info("Waiting for 5 seconds before next LLM evaluation call...")
             time.sleep(5)

    output_file_json = "pipeline_evaluation_results.json"
    with open(output_file_json, "w") as f:
        json.dump(all_evaluations, f, indent=2)
    logger.info(f"All evaluation results saved to {output_file_json}")

    output_file_text = "pipeline_evaluation_comments.txt"
    with open(output_file_text, "w") as f:
        f.write(evaluation_comments_text)
    logger.info(f"All evaluation comments saved to {output_file_text}")
    
    total_syntactic_score = 0
    query_review_count = 0
    for eval_item in all_evaluations:
        if "evaluation" in eval_item and \
           isinstance(eval_item["evaluation"], dict) and \
           "query_review" in eval_item["evaluation"] and \
           isinstance(eval_item["evaluation"]["query_review"], dict) and \
           "syntactic_validity" in eval_item["evaluation"]["query_review"] and \
           isinstance(eval_item["evaluation"]["query_review"]["syntactic_validity"], (int, float)):
            total_syntactic_score += eval_item["evaluation"]["query_review"]["syntactic_validity"]
            query_review_count += 1

    if query_review_count > 0:
        avg_syntactic_score = total_syntactic_score / query_review_count
        logger.info(f"Average Syntactic Validity Score: {avg_syntactic_score:.2f} out of 5 (from {query_review_count} reviews)")
    else:
        logger.info("No valid query reviews found to calculate average syntactic validity.")
