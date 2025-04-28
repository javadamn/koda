# tasks.py
from typing import Dict, List
from crewai import Task, Agent

def define_analysis_tasks(agents: Dict[str, Agent], user_query: str) -> List[Task]:
    """Defines the sequence of tasks for the microbial analysis crew."""

    # Ensure agents exist
    query_constructor = agents.get("query_constructor")
    information_retriever = agents.get("information_retriever")
    contextual_analyzer = agents.get("contextual_analyzer")
    report_writer = agents.get("report_writer")

    if not all([query_constructor, information_retriever, contextual_analyzer, report_writer]):
        raise ValueError("One or more required agents are missing.")

    tasks = []
    # Task 1: Construct Query
    tasks.append(
        Task(
            description=f"""
                1. Analyze the user query: '{user_query}'
                2. Consult the graph schema (provided in your goal or use the schema tool if needed).
                3. Formulate the optimal Cypher query, ensuring case-insensitive matching for names using `toLower()`.
                4. Output the query and any parameters as a JSON string, ready for the 'Execute Cypher Query Tool'.
                """,
            expected_output="A JSON string containing the 'query' and 'params' keys (e.g., '{\"query\": \"MATCH ...\", \"params\": {...}}').",
            agent=query_constructor,
            # human_input=True # Optional: Add human review step for the query
        )
    )

    # Task 2: Retrieve Data
    tasks.append(
        Task(
            description="""
                1. Take the JSON string output from the previous task (containing query and params).
                2. Prepare the input for the 'Execute Cypher Query Tool' by wrapping the JSON string under the 'query_info' key.
                3. Execute the tool.
                4. Output the raw results (list of dictionaries) or the error dictionary returned by the tool.
                """,
            expected_output="A list of dictionaries representing the query results, or a dictionary containing an 'error' key.",
            agent=information_retriever,
            context=[tasks[-1]] # Depends on the query output
        )
    )

    # Task 3: Analyze Results
    tasks.append(
        Task(
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
            agent=contextual_analyzer,
            context=[tasks[-1]] # Depends on the retrieved data
        )
    )

    # Task 4: Write Report
    tasks.append(
        Task(
            description="""
                1. Take the textual analysis from the previous task.
                2. Synthesize the key findings into a concise and well-structured final report.
                3. Ensure the report directly addresses the original user query.
                4. Format the report clearly (e.g., using markdown).
                """,
            expected_output="A final, formatted report summarizing the analysis and answering the user query.",
            agent=report_writer,
            context=[tasks[-1]] # Depends on the analysis
        )
    )

    return tasks