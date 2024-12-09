import streamlit as st
from llm import llm, embeddings
from graph import graph

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# First, create the vector index if it doesn't exist
try:
    # Try to use existing index
    neo4jvector = Neo4jVector.from_existing_index(
        embeddings,
        graph=graph,
        index_name="moviePlots",
        node_label="Movie",
        text_node_property="plot",
        embedding_node_property="plotEmbedding",
        retrieval_query="""
    RETURN
        node.plot AS text,
        score,
        {
            title: node.title,
            directors: [ (person)-[:DIRECTED]->(node) | person.name ],
            actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
            tmdbId: node.tmdbId,
            source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
        } AS metadata
    """
    )
except ValueError:
    # If index doesn't exist, create it
    neo4jvector = Neo4jVector.from_texts(
        [],  # No initial texts
        embeddings,
        index_name="moviePlots",
        node_label="Movie",
        embedding_node_property="plotEmbedding",
        text_node_property="plot",
        graph=graph,
        retrieval_query="""
    RETURN
        node.plot AS text,
        score,
        {
            title: node.title,
            directors: [ (person)-[:DIRECTED]->(node) | person.name ],
            actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
            tmdbId: node.tmdbId,
            source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
        } AS metadata
    """
    )
    # Create the index
    graph.query("""
    CREATE VECTOR INDEX moviePlots IF NOT EXISTS
    FOR (m:Movie)
    ON (m.plotEmbedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 1536,
        `vector.similarity_function`: 'cosine'
    }}
    """)

retriever = neo4jvector.as_retriever()

instructions = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
plot_retriever = create_retrieval_chain(retriever, question_answer_chain)

def get_movie_plot(input):
    return plot_retriever.invoke({"input": input})