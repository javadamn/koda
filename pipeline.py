from crewai import Agent, Task, Crew
from tools import analyze_microbial_interactions
from knowledge_graph import Neo4jKnowledgeGraph
import logging
from crewai.tools import tool




class MicrobialAnalysisPipeline:
    def __init__(self):
        self.agents = {
            "analyzer": Agent(
                role="Microbial Community Analyst",
                goal="Analyze microbial community interactions and generate a report based on the user's query.",
                backstory="Expert in microbial ecology, proficient in using a knowledge graph to understand microbe-metabolite relationships and their implications.",
                tools=[analyze_microbial_interactions],
                verbose=True
            ),
            "reporter": Agent( #added
                role="Report Writer",
                goal="Generate a concise and informative report.",
                backstory="Experienced scientific communicator, skilled at summarizing complex information.",
                tools=[],
                verbose=True
            )
        }

    def run_analysis(self, query: str) -> str:
        """Runs the microbial community analysis pipeline for a given user query."""
        analyzer_agent = self.agents["analyzer"]
        reporter_agent = self.agents["reporter"] #added

        task_analysis = Task(
            description=f"Analyze the microbial community based on the user query: '{query}'.  Focus on identifying key microbial interactions and their potential implications.  The output should be a dictionary.",
            expected_output="A dictionary containing a report summarizing the analysis.",
            agent=analyzer_agent
        )

        task_report = Task( #added
            description="Generate a concise report of the analysis.",
            expected_output="A short, informative report.",
            agent=reporter_agent,
            context=[task_analysis] #takes the output of the analysis task
        )

        crew = Crew(
            agents=[analyzer_agent, reporter_agent], #added reporter_agent
            tasks=[task_analysis, task_report], #added task_report
            verbose=True
        )
        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            error_message = f"Error running the crew: {e}"
            logger.error(error_message)
            return error_message

