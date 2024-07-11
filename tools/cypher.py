from llm import llm
from graph import graph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate


def create_cypher_prompt_template():
    """
    Creates a prompt template for Cypher queries.

    Returns:
        PromptTemplate: The Cypher prompt template
    """
    return PromptTemplate.from_template("""
        You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies 
        and provide recommendations.
        Convert the user's question based on the schema.
        
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided.
        
        Do not return entire nodes or embedding properties.
        
        Instructions:
        For movie titles that begin with "The", move "The" to the end of the title and use this new title in the query. 
        For example "The 39 Steps" becomes "39 Steps, The" or "The Matrix" becomes "Matrix, The".
        
        Example Cypher Queries:
        1. To find who acted in a movie:
        ```
        MATCH (p:Person)-[r:ACTED_IN]->(m:Movie {{title: "Movie Title"}})
        RETURN p.name, r.role
        ```
        
        2. To find who directed a movie:
        ```
        MATCH (p:Person)-[r:DIRECTED]->(m:Movie {{title: "Movie Title"}})
        RETURN p.name
        ```
        
        3. To find the degrees of separation between actors
        ```
        MATCH path = shortestPath((p:Person)-[:ACTED_IN|DIRECTED*]-(p:Person))
        RETURN length(path) as degrees_of_separation
        ```
        
        Schema:
        {schema}
        
        Question:
        {question}
        
        Cypher Query:""")


def create_cypher_qa_chain(prompt_template: PromptTemplate) -> GraphCypherQAChain:
    """
    Creates a GraphCypherQAChain using the provided prompt template and Neo4j graph.

    Args:
        prompt_template (PromptTemplate): The prompt template for Cypher queries
    """
    return GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        cypher_prompt=prompt_template,
        return_embedding=False,  # Don't return embedding properties for better performance
    )


cypher_qa = create_cypher_qa_chain(prompt_template=create_cypher_prompt_template())
