from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from typing import List

from microflow.tools.custom_tool import Neo4jTool

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class CompbioCrew:
    """Compbio crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],
            tools=[SerperDevTool(), Neo4jTool()],
            tool_choice="auto",
            verbose=True,  # type: ignore[index]
        )

    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_engineer"],  # type: ignore[index]
            tool_choice="auto",
            tools=[Neo4jTool()],
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def content_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config["content_reviewer"],  # type: ignore[index]
            tools=[Neo4jTool()],
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["reporting_analyst"],  # type: ignore[index]
            allow_delegation=False,
            verbose=True,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def create_query_task(self) -> Task:
        return Task(
            config=self.tasks_config["create_query_task"],  # type: ignore[index]
            verbose=True,
        )

    @task
    def execute_query_task(self) -> Task:
        return Task(
            config=self.tasks_config["execute_query_task"],  # type: ignore[index]
            context=[self.create_query_task()],
            tools=[Neo4jTool()],
            verbose=True,
        )

    @task
    def analyze_retrieved_data_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_retrieved_data_task"],  # type: ignore[index]
            context=[self.execute_query_task()],
            verbose=True,
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],  # type: ignore[index]
            context=[self.analyze_retrieved_data_task()],
            output_file="report.md",
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Compbio crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
