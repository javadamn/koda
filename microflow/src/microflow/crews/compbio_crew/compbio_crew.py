from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from typing import List

from microflow.tools.neo4j_tool import (
    Neo4jExecuteCypherQueryTool,
    Neo4jGetGraphSchemaTool,
)

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
            tools=[SerperDevTool(), Neo4jExecuteCypherQueryTool()],
            tool_choice="auto",
            verbose=True,  # type: ignore[index]
        )

    @agent
    def data_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["data_engineer"],  # type: ignore[index]
            tool_choice="auto",
            tools=[Neo4jGetGraphSchemaTool(), Neo4jExecuteCypherQueryTool()],
            memory=True,
            allow_delegation=False,
            verbose=True,
        )

    @agent
    def content_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["content_analyst"],  # type: ignore[index]
            tools=[],
            allow_delegation=False,
            verbose=True,
            memory=True,
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writer"],  # type: ignore[index]
            allow_delegation=False,
            verbose=True,
            memory=False,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def create_and_execute_cypher_query(self) -> Task:
        return Task(
            config=self.tasks_config["create_and_execute_cypher_query"],  # type: ignore[index]
            agent=self.data_engineer(),
            verbose=True,
        )

    # @task
    # def execute_query_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["execute_query_task"],  # type: ignore[index]
    #         context=[self.create_query_task()],
    #         tools=[Neo4jTool()],
    #         verbose=True,
    #     )

    @task
    def analyze_retrieved_data(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_retrieved_data"],  # type: ignore[index]
            agent=self.content_analyst(),
            context=[self.create_and_execute_cypher_query()],
            verbose=True,
        )

    @task
    def write_report(self) -> Task:
        return Task(
            config=self.tasks_config["write_report"],  # type: ignore[index]
            context=[self.analyze_retrieved_data()],
            agent=self.report_writer(),
            output_file="report.md",
            verbose=True,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Compbio crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            name="Compbio Crew",
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
