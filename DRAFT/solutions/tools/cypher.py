import streamlit as st

# tag::import[]
from langchain.chains import GraphCypherQAChain

from llm import llm
from graph import graph
# end::import[]


# tag::cypher-qa[]
cypher_qa = GraphCypherQAChain.from_llm(
    llm,          # <1>
    graph=graph,  # <2>
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


"""
The `kg_qa` can now be registered as a tool within the agent.

# tag::importcypherqa[]
from tools.cypher import cypher_qa
# end::importcypherqa[]

# tag::tool[]
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
        ),
    Tool.from_function(
        name="Vector Search Index",
        description="Provides information about movie plots using Vector Search",
        func = kg_qa,
        return_direct=True
    ),
    Tool.from_function(
        name="Graph Cypher QA Chain",  # <1>
        description="Provides information about Movies including their Actors, Directors and User reviews", # <2>
        func = cypher_qa, # <3>
        return_direct=True
    ),
]
# end::tool[]
"""