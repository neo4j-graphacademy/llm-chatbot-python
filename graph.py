import os

from langchain_community.graphs import Neo4jGraph
import streamlit as st


def connect_to_neo4j() -> Neo4jGraph:
    """
    Connects to a Neo4j graph using the provided credentials.

    Returns:
        Neo4jGraph: The Neo4j graph instance
    """
    if not os.environ.get('NEO4J_URI'):
        # return ()
        return Neo4jGraph(
            url=st.secrets["NEO4J_URI"],
            username=st.secrets["NEO4J_USERNAME"],
            password=st.secrets["NEO4J_PASSWORD"]
        )
    else:
        return Neo4jGraph(
            url=os.environ.get("NEO4J_URI"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD")
        )


# Connect to Neo4j and provide the graph as
# from graph import graph
graph = connect_to_neo4j()
