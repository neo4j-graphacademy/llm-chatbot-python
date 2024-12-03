import streamlit as st
from llm import llm
from graph import graph

# tag::import[]
from langchain_neo4j import GraphCypherQAChain
# end::import[]

# tag::cypher-qa[]
cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)
# end::cypher-qa[]
