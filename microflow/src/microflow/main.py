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
import logging

logger = logging.getLogger(__name__)


class DataQuestionFlowState(BaseModel):
    question: str = ""
    answer: str = ""
    iteration: int = 0
    status: str = ""
    is_answer_confirmed: bool = False


class DataQuestionFlow(Flow[DataQuestionFlowState]):

    NEED_TO_LOAD_KNOWLEDGE_GRAPH: bool = True

    @start()
    def get_user_question(self):
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
        os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
        os.environ["NEO4J_USER"] = os.getenv("NEO4J_USER")
        os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")
        os.environ["NEO4J_KB_PATH"] = os.getenv("NEO4J_KB_PATH", "None")

        if os.getenv("NEO4J_KB_PATH") != "None" and self.NEED_TO_LOAD_KNOWLEDGE_GRAPH:
            from microflow.tools.neo4j_tool import Neo4jLoadKnowledgeGraphTool

            print("Loading knowledge base...")
            Neo4jLoadKnowledgeGraphTool.NEO4J_URI = os.getenv("NEO4J_URI")
            Neo4jLoadKnowledgeGraphTool.NEO4J_USER = os.getenv("NEO4J_USER")
            Neo4jLoadKnowledgeGraphTool.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
            Neo4jLoadKnowledgeGraphTool.get_driver(reset=True)
            Neo4jLoadKnowledgeGraphTool._run(path=os.getenv("NEO4J_KB_PATH"))

            self.NEED_TO_LOAD_KNOWLEDGE_GRAPH = False
            logger.info("Knowledge base loaded successfully")

        print("Waiting for user question...")

        self.state.question = input(
            "Enter your question about the microbiome knowledgebase: "
        )
        self.state.status = "Question received"

        return self.state

    @listen(get_user_question)
    def process_question(self, state):
        print("Working on an answer to the question...")
        topic = "Microbial Ecology"  # TODO: remove hardcoded topic
        inputs = {
            "question": self.state.question,
            "topic": topic,
            "GRAPH_SCHEMA_DESCRIPTION": MICROBIOME_GRAPH_SCHEMA_DESCRIPTION,
        }

        print("Running Compbio Crew with the following inputs: {")
        for key, value in inputs.items():
            print(f"{key}: {value}")
        print("}")
        print("--------------------------------")

        result = CompbioCrew().crew().kickoff(inputs=inputs)

        self.state.status = "Answer generated"
        self.state.answer = result
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
