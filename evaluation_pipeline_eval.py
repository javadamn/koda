# evaluation_pipeline.py
import json
import time
import re
import os # Added for environment variables
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, AliasChoices, ConfigDict, ValidationError
from crewai import Agent, Task, Crew, Process, TaskOutput

# Assuming get_llm is adapted in llm_setup.py or similar
from llm_setup import get_llm as get_llm_for_pipeline # Rename to avoid conflict if needed, or use same
# Or define a specific one for evaluators if needed (as done below)

from pipeline1_eval import MicrobialAnalysisPipeline
from schema import GRAPH_SCHEMA_DESCRIPTION
import config

logger = config.get_logger("EvaluationPipeline")

# --- Define your list of LLMs for evaluation ---
# !!! REPLACE THESE WITH YOUR ACTUAL VALID MODEL IDENTIFIERS !!!
# Ensure your get_eval_llm function and API keys can handle these.
LLM_LIST_FOR_EVALUATION = [
    "gpt-4o",  # OpenAI
    "gpt-3.5-turbo",

    # "claude-3-opus-20240229", # Anthropic
    # "gemini-1.5-pro-latest",   # Google Vertex AI
    # "deepseek-chat" # DeepSeek (ensure this model name is correct for your DeepSeek setup)
    # Example for DeepSeek if using OpenAI-compatible endpoint: "deepseek/deepseek-chat" or just "deepseek-chat"
]

if len(LLM_LIST_FOR_EVALUATION) < 4:
    logger.warning(f"Expected 4 LLMs in LLM_LIST_FOR_EVALUATION for full setup, but found {len(LLM_LIST_FOR_EVALUATION)}. Adjusting logic.")
    # The script will still run but might not have 3 distinct reviewers for each generator if the list is too small.

EVAL_LLM_DEFAULT_TEMPERATURE = 0.1


# --- Pydantic Models (CypherReviewSchema, AnalysisReviewSchema) ---
# ... (These remain the same as your last working version) ...
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


# --- LLM and Agent Factory Functions ---
def get_eval_llm(model_name: str, temperature: float = EVAL_LLM_DEFAULT_TEMPERATURE):
    """
    Wrapper to get LLM for evaluation, using the more comprehensive get_llm from llm_setup.
    This assumes get_llm in llm_setup can handle all providers.
    """
    return get_llm_for_pipeline(model_name=model_name, temperature=temperature)


def create_cypher_reviewer_agent(llm_instance):
    # ... (Keep your existing detailed prompt for CypherReviewSchema) ...
    return Agent(
        role="Expert Cypher Query Reviewer and Neo4j Specialist",
        goal=f"""
            Critically evaluate a given Cypher query based on multiple criteria.
            You MUST provide your evaluation as a valid JSON object that strictly conforms to the CypherReviewSchema.
            Ensure all JSON keys are exactly as defined in the schema (e.g., 'syntactic_validity', 'schema_adherence').
            The following fields MUST be integers on a scale of 1 (poor) to 5 (excellent):
            - 'syntactic_validity'
            - 'schema_adherence'
            - 'semantic_accuracy_nlq'
            - 'semantic_accuracy_gold' (if applicable, provide an integer 1-5, otherwise use null or omit this field if not applicable)
            - 'parameterization'
            - 'tolower_usage'
            The 'qualitative_feedback' field MUST be a string containing your detailed comments.
            The 'is_executable_in_neo4j' field MUST be a boolean (true/false), or use null or omit this field if unknown.
            Your entire response should be ONLY the JSON object, without any surrounding text, thoughts, or markdown backticks.
            GRAPH SCHEMA for reference:
            ---
            {GRAPH_SCHEMA_DESCRIPTION}
            ---
        """,
        backstory="You are a meticulous Neo4j expert...",
        llm=llm_instance, verbose=True, allow_delegation=False,
        output_json_parser=CypherReviewSchema
    )

def create_analysis_reviewer_agent(llm_instance):
    # ... (Keep your existing detailed prompt for AnalysisReviewSchema) ...
    return Agent(
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
            - 'drug_target_discussion_quality' (if applicable, provide an integer 1-5, otherwise use null or omit this field if not applicable)
            The 'qualitative_feedback' field MUST be a string containing your detailed comments.
            Your entire response should be ONLY the JSON object, without any surrounding text, thoughts, or markdown backticks.
        """,
        backstory="You are a seasoned researcher in microbial genomics...",
        llm=llm_instance, verbose=True, allow_delegation=False,
        output_json_parser=AnalysisReviewSchema
    )

# --- Helper functions _get_agent_final_answer_str_from_task_output_eval and _extract_and_validate_json_review ---
# ... (These should be the latest working versions from your previous iteration,
# Ensure _extract_and_validate_json_review correctly handles the 'raw' key from TaskOutput string representations)
def _get_agent_final_answer_str_from_task_output_eval(task_output_obj: Any) -> Optional[str]:
    if task_output_obj is None: return None
    
    if isinstance(task_output_obj, BaseModel):
        try: return task_output_obj.model_dump_json()
        except Exception: return str(task_output_obj)
    if isinstance(task_output_obj, dict):
        try: return json.dumps(task_output_obj) 
        except Exception: pass 
    if isinstance(task_output_obj, str): return task_output_obj

    text_to_search = ""
    if isinstance(task_output_obj, TaskOutput):
        if task_output_obj.exported_output is not None:
            if isinstance(task_output_obj.exported_output, str): return task_output_obj.exported_output.strip()
            if isinstance(task_output_obj.exported_output, dict): return json.dumps(task_output_obj.exported_output)
            if isinstance(task_output_obj.exported_output, BaseModel): return task_output_obj.exported_output.model_dump_json()
        
        if hasattr(task_output_obj, 'raw') and isinstance(task_output_obj.raw, str) and task_output_obj.raw.strip():
            text_to_search = task_output_obj.raw
        elif task_output_obj.description and isinstance(task_output_obj.description, str):
            text_to_search = task_output_obj.description
        else: text_to_search = str(task_output_obj)
    else:
        text_to_search = str(task_output_obj)

    logger.debug(f"_get_agent_final_answer_str_from_task_output_eval: Text to search (first 300 chars): {text_to_search[:300]}")
    
    json_block_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text_to_search, re.IGNORECASE | re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()
    
    return text_to_search.strip()


def _extract_and_validate_json_review(task_output_obj: Any, pydantic_model: type[BaseModel]) -> Dict[str, Any]:
    if isinstance(task_output_obj, pydantic_model):
        logger.info(f"Review output is already a validated Pydantic model: {pydantic_model.__name__}")
        return task_output_obj.model_dump()
    
    agent_final_answer_str = _get_agent_final_answer_str_from_task_output_eval(task_output_obj)

    if not agent_final_answer_str:
        logger.error(f"Could not extract a string from task_output_obj for {pydantic_model.__name__}.")
        return {"error": f"No parsable string output from task for {pydantic_model.__name__}.", "raw_task_output": str(task_output_obj)}

    logger.info(f"Attempting to parse and validate review output for {pydantic_model.__name__} (from agent's final answer string): {agent_final_answer_str[:500]}...")
    
    try:
        parsed_outer_dict = json.loads(agent_final_answer_str)
        actual_review_dict_to_validate = None

        if isinstance(parsed_outer_dict, dict):
            if 'raw' in parsed_outer_dict and isinstance(parsed_outer_dict['raw'], str) and 'agent' in parsed_outer_dict :
                json_str_from_raw_key = parsed_outer_dict['raw']
                logger.info(f"Detected TaskOutput-like structure. Extracting review JSON from 'raw' key for {pydantic_model.__name__}: {json_str_from_raw_key[:300]}...")
                try:
                    actual_review_dict_to_validate = json.loads(json_str_from_raw_key)
                except json.JSONDecodeError as e_inner:
                    logger.error(f"JSONDecodeError parsing content of 'raw' key for {pydantic_model.__name__}: {e_inner}. String from 'raw': {json_str_from_raw_key}")
                    return {"error": f"Failed to parse JSON from 'raw' key for {pydantic_model.__name__}.", "raw_key_content": json_str_from_raw_key, "original_agent_output_string": agent_final_answer_str}
            else:
                logger.info(f"Parsed agent's final answer string directly into a dictionary for {pydantic_model.__name__}. Assuming this is the review content.")
                actual_review_dict_to_validate = parsed_outer_dict
        else: # If agent_final_answer_str itself was the direct review JSON, not a TaskOutput string representation
            logger.info(f"Agent's final answer string was not a dict after initial parse. Assuming it IS the direct review JSON for {pydantic_model.__name__}.")
            actual_review_dict_to_validate = json.loads(agent_final_answer_str) # This was the original intent if string is pure JSON

        if actual_review_dict_to_validate is not None and isinstance(actual_review_dict_to_validate, dict):
            model_instance = pydantic_model.model_validate(actual_review_dict_to_validate) # This uses aliases
            logger.info(f"Successfully validated JSON against {pydantic_model.__name__}.")
            return model_instance.model_dump()
        elif actual_review_dict_to_validate is not None: # Parsed to non-dict
             logger.error(f"Isolated review content for {pydantic_model.__name__} is not a dictionary. Type: {type(actual_review_dict_to_validate)}. Content: {str(actual_review_dict_to_validate)[:300]}")
             return {"error": f"Isolated review content for {pydantic_model.__name__} is not a dict.", "raw_content": str(actual_review_dict_to_validate)}
        else:
            logger.error(f"Could not determine the actual review dictionary to validate for {pydantic_model.__name__}.")
            return {"error": f"Could not isolate actual review dictionary for {pydantic_model.__name__}.", "raw_original_agent_str": agent_final_answer_str}

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError parsing agent_final_answer_str for {pydantic_model.__name__}: {e}. String used: {agent_final_answer_str[:500]}")
        return {"error": f"Failed to parse agent_final_answer_str as JSON for {pydantic_model.__name__}.", "raw": agent_final_answer_str}
    except ValidationError as ve:
        dict_being_validated_str = str(actual_review_dict_to_validate if 'actual_review_dict_to_validate' in locals() and actual_review_dict_to_validate is not None else agent_final_answer_str)
        logger.error(f"Pydantic ValidationError for {pydantic_model.__name__}: {ve}. Content attempted for validation: {dict_being_validated_str[:500]}")
        return {"error": f"Pydantic validation failed for {pydantic_model.__name__}.", "raw_content_validated": dict_being_validated_str, "validation_error_details": str(ve)}
    except Exception as e_unexp:
        logger.error(f"Unexpected error in _extract_and_validate_json_review for {pydantic_model.__name__}: {e_unexp}. String used: {agent_final_answer_str[:500]}", exc_info=True)
        return {"error": f"Unexpected error during review validation for {pydantic_model.__name__}.", "raw": agent_final_answer_str}

# --- Main Evaluation Function (Modified) ---
def evaluate_pipeline_output(
    nlq: str,
    generated_query_json_str: str,
    retrieved_data: Union[List[Dict[str, Any]], Dict[str, Any]],
    generated_report: str,
    reviewer_llm_names: List[str],
    gold_standard_cypher: Optional[str] = None,
) -> Dict[str, Any]:
    # ... (This function remains largely the same as the one provided in the previous step,
    # ensure it correctly calls create_cypher_reviewer_agent and create_analysis_reviewer_agent
    # with LLM instances derived from reviewer_llm_names and get_eval_llm)
    per_reviewer_evaluations = {}
    parsed_query_dict_for_reviewer = {} 

    if not generated_query_json_str or not isinstance(generated_query_json_str, str):
        logger.error(f"Invalid generated_query_json_str for NLQ '{nlq}': Type {type(generated_query_json_str)}, Value: {generated_query_json_str}")
        parsed_query_dict_for_reviewer = {"error": "Invalid or missing generated Cypher JSON string from main pipeline."}
    else:
        try:
            parsed_query_dict_for_reviewer = json.loads(generated_query_json_str)
            if "error" in parsed_query_dict_for_reviewer: 
                 logger.error(f"The 'generated_query_json_str' from main pipeline contained an error: {parsed_query_dict_for_reviewer['error']}")
        except json.JSONDecodeError:
            logger.error(f"Could not parse generated_query_json_str for NLQ '{nlq}': {generated_query_json_str}")
            parsed_query_dict_for_reviewer = {"error": "Generated Cypher by main pipeline is not valid JSON.", "raw_query_str": generated_query_json_str}

    for reviewer_llm_name in reviewer_llm_names:
        logger.info(f"--- Evaluating with Reviewer LLM: {reviewer_llm_name} for NLQ: '{nlq}' ---")
        try:
            reviewer_llm_instance = get_eval_llm(model_name=reviewer_llm_name)
        except Exception as e_llm_init:
            logger.error(f"Failed to initialize reviewer LLM {reviewer_llm_name}: {e_llm_init}", exc_info=True)
            per_reviewer_evaluations[reviewer_llm_name] = {
                "query_review": {"error": f"Failed to initialize reviewer LLM {reviewer_llm_name}."},
                "analysis_review": {"error": f"Failed to initialize reviewer LLM {reviewer_llm_name}."}
            }
            continue
            
        current_cypher_reviewer_agent = create_cypher_reviewer_agent(reviewer_llm_instance)
        current_analysis_reviewer_agent = create_analysis_reviewer_agent(reviewer_llm_instance)

        query_review_result = {}
        query_review_task_description = "" 
        
        if "error" not in parsed_query_dict_for_reviewer:
            # ... (setup query_review_task_description as before) ...
            query_review_task_description = f"""
                Natural Language Question (NLQ): "{nlq}"
                Generated Cypher Query to review:
                Query: "{parsed_query_dict_for_reviewer.get('query', 'QUERY STRING NOT FOUND IN PARSED JSON')}"
                Params: {json.dumps(parsed_query_dict_for_reviewer.get('params', {}))}
            """
            if gold_standard_cypher:
                query_review_task_description += f"\nGold Standard Cypher Query (for semantic comparison):\n{gold_standard_cypher}"
            query_review_task_description += """
                Based on the NLQ, the generated query, the (optional) gold standard query, and the GRAPH SCHEMA provided in your role,
                evaluate the generated Cypher query according to all criteria mentioned in your goal.
                Your output MUST be a valid JSON object conforming to the CypherReviewSchema, using snake_case for all keys.
            """
            query_review_task = Task(
                description=query_review_task_description,
                expected_output="A JSON object conforming to CypherReviewSchema.",
                agent=current_cypher_reviewer_agent,
            )
        else:
            query_review_task = None 
            query_review_result = parsed_query_dict_for_reviewer 
            logger.info(f"Skipping Cypher query review with {reviewer_llm_name} for NLQ '{nlq}' due to error in generated query from main pipeline.")

        # ... (setup retrieved_data_for_llm_str and analysis_review_task_description as before) ...
        retrieved_data_for_llm_str = ""
        if isinstance(retrieved_data, dict) and "error" in retrieved_data:
            retrieved_data_for_llm_str = f"Data Retrieval Error from Main Pipeline: {retrieved_data['error']}"
            if "raw" in retrieved_data: 
                 retrieved_data_for_llm_str += f" Raw content from main pipeline: {str(retrieved_data['raw'])[:200]}"
        elif isinstance(retrieved_data, list):
            retrieved_data_for_llm_str = json.dumps(retrieved_data[:5], indent=2) 
            if len(retrieved_data) > 5:
                retrieved_data_for_llm_str += f"\n... and {len(retrieved_data) - 5} more records."
        elif retrieved_data is None:
            retrieved_data_for_llm_str = "No data was retrieved by the main pipeline (retrieved_data is None)."
        else:
            retrieved_data_for_llm_str = f"Retrieved data from main pipeline is in an unexpected format: {type(retrieved_data)}"

        analysis_review_task_description = f"""
            Natural Language Question (NLQ): "{nlq}"
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
            agent=current_analysis_reviewer_agent,
        )

        tasks_to_run_this_reviewer = []
        if query_review_task: tasks_to_run_this_reviewer.append(query_review_task)
        tasks_to_run_this_reviewer.append(analysis_review_task) 
        
        if not tasks_to_run_this_reviewer: 
            logger.error(f"No evaluation tasks to run for reviewer {reviewer_llm_name}.")
            # Ensure query_review_result is initialized if query_review_task was None
            if not query_review_result: query_review_result = {"error": "Query review task not run."}
            per_reviewer_evaluations[reviewer_llm_name] = {
                "query_review": query_review_result,
                "analysis_review": {"error": "No analysis review task was run."}
            }
            continue

        evaluation_crew = Crew(
            agents=[current_cypher_reviewer_agent, current_analysis_reviewer_agent],
            tasks=tasks_to_run_this_reviewer,
            process=Process.sequential,
            verbose=1 
        )

        logger.info(f"Kicking off evaluation crew with reviewer: {reviewer_llm_name} for NLQ: '{nlq}'")
        evaluation_crew.kickoff() 

        if not query_review_result: 
            query_review_result = _extract_and_validate_json_review(
                query_review_task.output if query_review_task else None, CypherReviewSchema
            )
        
        analysis_review_result = _extract_and_validate_json_review(
            analysis_review_task.output, AnalysisReviewSchema
        )
        
        per_reviewer_evaluations[reviewer_llm_name] = {
            "query_review": query_review_result,
            "analysis_review": analysis_review_result
        }
        logger.info(f"--- Evaluation finished with Reviewer LLM: {reviewer_llm_name} for NLQ: '{nlq}' ---")
        time.sleep(config.get_value('EVAL_DELAY_BETWEEN_REVIEWERS', 2))

    return per_reviewer_evaluations


# --- Main Execution Block (remains the same as your last working multi-LLM version) ---
if __name__ == "__main__":
    if not LLM_LIST_FOR_EVALUATION or len(LLM_LIST_FOR_EVALUATION) == 0:
        logger.error("LLM_LIST_FOR_EVALUATION is not defined or empty. Please define at least one LLM.")
        exit()
    
    # ... (The rest of your __main__ block from the previous comprehensive answer
    #      that handles iterating through generators, calling evaluate_pipeline_output,
    #      collecting results, calculating averages, and saving to files,
    #      should largely remain the same.)
    # --- Main Execution Block (from previous good version, ensure it's complete) ---
    # LLM_LIST_FOR_EVALUATION is defined at the top of the file

    benchmark_items = [
        {
            "nlq": "What KEGG Orthologies (KOs) are associated with the microbe Bifidobacterium_longum_longum_JDM301 and what are their functional descriptions?",
            "gold_standard_cypher": "MATCH (m:microbe)-[r:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(m.name) = toLower('Bifidobacterium_longum_longum_JDM301') RETURN k.name AS ko_id, r.description AS ko_functional_description"
        },
        {
            "nlq": "Which microbes produce Thiamine and also have KOs whose description mentions 'synthase'?",
            "gold_standard_cypher": "MATCH (m:microbe)-[:PRODUCES]->(met:metabolite) WHERE toLower(met.name) = toLower('Thiamine') WITH m MATCH (m)-[r_ko:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(r_ko.description) CONTAINS toLower('synthase') RETURN DISTINCT m.name AS microbe_name"
        },
        {
            "nlq": "How many distinct KOs are associated with Streptococcus_pneumoniae_G54?",
            "gold_standard_cypher": "MATCH (m:microbe)-[:HAS_KEGG_ORTHOLOGY]->(k:KO) WHERE toLower(m.name) = toLower('Streptococcus_pneumoniae_G54') RETURN count(DISTINCT k.name) AS distinct_ko_count"
        }
    ]

    overall_results_by_generator = {} 

    for generator_llm_model_name_for_pipeline in LLM_LIST_FOR_EVALUATION: # Renamed for clarity
        logger.info(f"\n\n{'='*20} USING GENERATOR LLM: {generator_llm_model_name_for_pipeline} {'='*20}")
        
        try:
            main_pipeline = MicrobialAnalysisPipeline(generator_llm_model_name=generator_llm_model_name_for_pipeline)
            logger.info(f"Main MicrobialAnalysisPipeline initialized with generator: {generator_llm_model_name_for_pipeline}.")
        except Exception as e:
            logger.error(f"Failed to initialize MicrobialAnalysisPipeline with {generator_llm_model_name_for_pipeline}: {e}", exc_info=True)
            overall_results_by_generator[generator_llm_model_name_for_pipeline] = {
                "error": f"Failed to initialize main pipeline with {generator_llm_model_name_for_pipeline}: {str(e)}",
                "nlq_evaluations": []
            }
            continue

        current_reviewer_llm_names = [name for name in LLM_LIST_FOR_EVALUATION if name != generator_llm_model_name_for_pipeline]
        if not current_reviewer_llm_names:
            if len(LLM_LIST_FOR_EVALUATION) == 1:
                logger.info(f"Only one LLM specified. Using {generator_llm_model_name_for_pipeline} as both generator and sole reviewer.")
                current_reviewer_llm_names = [generator_llm_model_name_for_pipeline]
            else: # Should not happen if list has >1 distinct models
                logger.warning(f"No distinct reviewer LLMs available for generator {generator_llm_model_name_for_pipeline}, though multiple LLMs are in the list. This is unexpected. Skipping reviews for this generator.")
                overall_results_by_generator[generator_llm_model_name_for_pipeline] = {
                    "error": f"No distinct reviewers for generator {generator_llm_model_name_for_pipeline}",
                    "nlq_evaluations": []
                }
                continue
        
        logger.info(f"Reviewer LLMs for generator {generator_llm_model_name_for_pipeline}: {current_reviewer_llm_names}")

        nlq_evaluations_for_this_generator = []
        master_evaluation_comments_text = f"# Evaluation Report for Generator: {generator_llm_model_name_for_pipeline}\n"

        for item_idx, item in enumerate(benchmark_items):
            nlq = item["nlq"]
            logger.info(f"\n--- Evaluating NLQ {item_idx + 1}/{len(benchmark_items)} for Generator '{generator_llm_model_name_for_pipeline}': {nlq} ---")
            master_evaluation_comments_text += f"\n\n## NLQ: {nlq}\n"

            pipeline_outputs = main_pipeline.run_analysis(nlq)
            if not isinstance(pipeline_outputs, dict) or \
               not all(k in pipeline_outputs for k in ["generated_cypher_json_str", "retrieved_data", "final_report"]):
                logger.error(f"Main pipeline returned malformed output for NLQ: {nlq} with generator {generator_llm_model_name_for_pipeline}.")
                nlq_eval_data = {
                    "nlq": nlq, "error": "Malformed output from main pipeline.",
                    "pipeline_outputs": str(pipeline_outputs)[:500], "reviews_by_llm": {}
                }
                nlq_evaluations_for_this_generator.append(nlq_eval_data)
                master_evaluation_comments_text += f"ERROR: Malformed output from main pipeline: {str(pipeline_outputs)[:500]}\n"
                continue
            
            generated_cypher_json_str = pipeline_outputs.get("generated_cypher_json_str")
            retrieved_data = pipeline_outputs.get("retrieved_data")
            final_report = pipeline_outputs.get("final_report")

            master_evaluation_comments_text += f"**Generated Cypher (by {generator_llm_model_name_for_pipeline})**: `{generated_cypher_json_str}`\n"
            # master_evaluation_comments_text += f"**Retrieved Data**: (First 200 chars) `{str(retrieved_data)[:200]}...`\n" # Potentially very long
            master_evaluation_comments_text += f"**Final Report (by {generator_llm_model_name_for_pipeline})**: (First 100 chars) `{str(final_report)[:100]}...`\n\n"

            per_reviewer_evaluation_results = evaluate_pipeline_output(
                nlq=nlq,
                generated_query_json_str=str(generated_cypher_json_str),
                retrieved_data=retrieved_data,
                generated_report=str(final_report),
                reviewer_llm_names=current_reviewer_llm_names,
                gold_standard_cypher=item.get("gold_standard_cypher")
            )
            
            nlq_eval_data = {
                "nlq": nlq,
                "pipeline_outputs_from_main": pipeline_outputs,
                "reviews_by_llm": per_reviewer_evaluation_results
            }
            nlq_evaluations_for_this_generator.append(nlq_eval_data)

            master_evaluation_comments_text += "### Evaluation Summary for this NLQ:\n"
            all_query_reviews_for_nlq = []
            all_analysis_reviews_for_nlq = []

            for reviewer_name, review_set in per_reviewer_evaluation_results.items():
                master_evaluation_comments_text += f"\n#### Review by: {reviewer_name}\n"
                qr = review_set.get("query_review", {})
                master_evaluation_comments_text += "**Query Review:**\n"
                if isinstance(qr, dict) and "error" not in qr:
                    all_query_reviews_for_nlq.append(qr)
                    master_evaluation_comments_text += f"  - Feedback: {qr.get('qualitative_feedback', 'N/A')}\n"
                    scores = {k: v for k, v in qr.items() if isinstance(v, (int, float))}
                    master_evaluation_comments_text += f"  - Scores: {scores}\n"
                else:
                    master_evaluation_comments_text += f"  - Error: {qr.get('error', 'Unknown error')}\n  - Raw: `{str(qr.get('raw_parsed_content', qr.get('raw_key_content', qr.get('raw', 'N/A'))))[:200]}...`\n"

                ar = review_set.get("analysis_review", {})
                master_evaluation_comments_text += "**Analysis Review:**\n"
                if isinstance(ar, dict) and "error" not in ar:
                    all_analysis_reviews_for_nlq.append(ar)
                    master_evaluation_comments_text += f"  - Feedback: {ar.get('qualitative_feedback', 'N/A')}\n"
                    scores = {k: v for k, v in ar.items() if isinstance(v, (int, float))}
                    master_evaluation_comments_text += f"  - Scores: {scores}\n"
                else:
                    master_evaluation_comments_text += f"  - Error: {ar.get('error', 'Unknown error')}\n  - Raw: `{str(ar.get('raw_parsed_content', ar.get('raw_key_content', ar.get('raw', 'N/A'))))[:200]}...`\n"

            avg_query_scores = {}
            if all_query_reviews_for_nlq:
                score_keys = set(k for review in all_query_reviews_for_nlq for k in review if isinstance(review[k], (int, float)))
                for key in score_keys:
                    values = [review[key] for review in all_query_reviews_for_nlq if key in review and isinstance(review[key], (int, float))]
                    if values: avg_query_scores[key] = round(sum(values) / len(values), 2)
            
            avg_analysis_scores = {}
            if all_analysis_reviews_for_nlq:
                score_keys = set(k for review in all_analysis_reviews_for_nlq for k in review if isinstance(review[k], (int, float)))
                for key in score_keys:
                    values = [review[key] for review in all_analysis_reviews_for_nlq if key in review and isinstance(review[key], (int, float))]
                    if values: avg_analysis_scores[key] = round(sum(values) / len(values), 2)

            master_evaluation_comments_text += f"\n**Average Query Scores (Generator: {generator_llm_model_name_for_pipeline}, NLQ: \"{nlq[:30]}...\")**: {avg_query_scores}\n"
            master_evaluation_comments_text += f"**Average Analysis Scores (Generator: {generator_llm_model_name_for_pipeline}, NLQ: \"{nlq[:30]}...\")**: {avg_analysis_scores}\n"
            logger.info(f"Average Query Scores (Generator {generator_llm_model_name_for_pipeline}, NLQ \"{nlq[:30]}...\"): {avg_query_scores}")
            logger.info(f"Average Analysis Scores (Generator {generator_llm_model_name_for_pipeline}, NLQ \"{nlq[:30]}...\"): {avg_analysis_scores}")

            if item_idx < len(benchmark_items) - 1:
                 logger.info(f"Waiting {config.get_value('EVAL_DELAY_BETWEEN_NLQ', 5)}s before next NLQ for Generator {generator_llm_model_name_for_pipeline}...")
                 time.sleep(config.get_value('EVAL_DELAY_BETWEEN_NLQ', 5)) # From config or default

        overall_results_by_generator[generator_llm_model_name_for_pipeline] = {"nlq_evaluations": nlq_evaluations_for_this_generator}
        
        output_file_text_generator = f"pipeline_evaluation_comments_{generator_llm_model_name_for_pipeline.replace('/', '_').replace(':', '_')}.txt"
        with open(output_file_text_generator, "w") as f:
            f.write(master_evaluation_comments_text)
        logger.info(f"Evaluation comments for generator {generator_llm_model_name_for_pipeline} saved to {output_file_text_generator}")

        if generator_llm_model_name_for_pipeline != LLM_LIST_FOR_EVALUATION[-1]:
             delay_btw_generators = config.get_value('EVAL_DELAY_BETWEEN_GENERATORS', 10)
             logger.info(f"Waiting {delay_btw_generators}s before switching to next generator LLM...")
             time.sleep(delay_btw_generators)

    output_file_json_overall = "pipeline_evaluation_results_by_generator.json"
    with open(output_file_json_overall, "w") as f:
        # Enhanced default for json.dump to handle Pydantic models or other complex objects
        def custom_serializer(obj):
            if isinstance(obj, (TaskOutput, BaseModel)): # Add other non-serializable types if needed
                return str(obj) # Or a more specific serialization
            try:
                return obj.__dict__ # For other custom objects
            except AttributeError:
                return str(obj)
        json.dump(overall_results_by_generator, f, indent=2, default=custom_serializer)
    logger.info(f"All evaluation results by generator saved to {output_file_json_overall}")
    
    logger.info("Multi-LLM evaluation process completed.")