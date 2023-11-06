import streamlit as st

from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph

from llm import llm
from graph import graph
from prompts import CYPHER_QA_PROMPT

qa_prompt = CYPHER_QA_PROMPT


# tag::cypher-qa[]
cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True
)
# end::cypher-qa[]

# tag::generate-response[]
def generate_response(prompt):
    """
    Use the Neo4j recommendations dataset to provide
    context to the LLM when answering a question
    """

    # Handle the response
    response = cypher_qa.run(prompt)

    return response
# end::generate-response[]
