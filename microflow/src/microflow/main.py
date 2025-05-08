#!/usr/bin/env python
import os
from random import randint
from dotenv import load_dotenv

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from microflow.crews.compbio_crew.compbio_crew import CompbioCrew
from microflow.dbs.microbiome_graph import (
    MICROBIOME_GRAPH_SCHEMA_DESCRIPTION,
)


class DataQuestionFlowState(BaseModel):
    question: str = ""
    answer: str = ""
    iteration: int = 0
    status: str = ""
    is_answer_confirmed: bool = False


class DataQuestionFlow(Flow[DataQuestionFlowState]):

    @start()
    def get_user_question(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
        os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
        os.environ["NEO4J_USER"] = os.getenv("NEO4J_USER")
        os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
        os.environ["NEO4J_KB_PATH"] = os.getenv("NEO4J_KB_PATH", "None")

        if os.getenv("NEO4J_KB_PATH") != "None":
            from microflow.tools.custom_tool import Neo4jTool

            print("Loading knowledge base...")
            Neo4jTool.NEO4J_URI = os.getenv("NEO4J_URI")
            Neo4jTool.NEO4J_USER = os.getenv("NEO4J_USER")
            Neo4jTool.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
            Neo4jTool.get_driver(reset=True)
            Neo4jTool._run_load_knowledge(path=os.getenv("NEO4J_KB_PATH"))

        print("Waiting for user question...")

        self.state.question = input(
            "Enter your question about the microbiome knowledgebase: "
        )
        self.state.status = "Question received"

        return self.state

    @listen(get_user_question)
    def process_question(self, state):
        print("Working on an answer to the question...")
        topic = "gut microbiome"  # TODO: remove hardcoded topic
        inputs = {
            "question": self.state.question,
            "topic": topic,
            "GRAPH_SCHEMA_DESCRIPTION": MICROBIOME_GRAPH_SCHEMA_DESCRIPTION,
        }

        result = CompbioCrew().crew().kickoff(inputs=inputs)

        self.state.status = "Answer generated"
        self.state.answer = result.raw
        self.state.is_answer_confirmed = (
            True  # TODO: can add an external tester agent to do this :?
        )
        self.state.iteration += 1  # For now we're not iterating

        print(f"Iteration {self.state.iteration} complete")
        print(f"Answer: {self.state.answer}")
        print(f"Status: {self.state.status}")
        print(f"Is answer confirmed: {self.state.is_answer_confirmed}")


def kickoff():
    data_question_flow = DataQuestionFlow()
    data_question_flow.kickoff()
    print("\n=== Flow Complete ===")


def plot():
    data_question_flow = DataQuestionFlow()
    data_question_flow.plot("data_question_flow")
    print(f"Flow visualization saved to 'data_question_flow.html'")


if __name__ == "__main__":
    kickoff()
