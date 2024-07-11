from llm import llm, embeddings
from graph import graph
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def create_neo4j_vector() -> Neo4jVector:
    """
    Creates a Neo4jVector using the provided Neo4j graph and embeddings.

    Returns:
        Neo4jVector: The Neo4jVector instance
    """
    return Neo4jVector(
        embeddings,
        graph=graph,
        index_name="moviePlots",
        node_label="Movie",
        text_node_property="plot",
        embedding_node_property="plotEmbedding",
        retrieval_query="""
        RETURN node.plot AS text, score,
        {
            title: node.title,
            directors: [ (person)-[:DIRECTED]->(node) | person.name ],
            actors: [ (person)-[r:ACTED_IN]->(node) | [person.name, r.role] ],
            tmdbId: node.tmdbId,
            source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
        } AS metadata
        """
    )


def create_prompt() -> ChatPromptTemplate:
    """
    Creates a ChatPromptTemplate for the movie chat chain.

    Returns:
        ChatPromptTemplate: The ChatPromptTemplate instance
    """
    instructions = (
        "Use the given context to answer the question."
        "If you don't know the answer, say you don't know."
        "Context: {context}"
    )

    return ChatPromptTemplate.from_messages(
        (
            ("system", instructions),
            # ("system", "{context}"),
            ("human", "{input}")
        )
    )


# Create the chain
def create_my_retrieval_chain(prompt, retriever):
    """
    Creates a RetrievalChain using the provided Neo4jVector and prompt.

    Returns:
        RetrievalChain: The RetrievalChain instance
    """

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(
        retriever,
        question_answer_chain,
        # chain_type="vectorstore",
        # retriever_type="dense_retriever_approximate",
        # return_source_documents=True,
        # k=5,
        # max_seq_len_increase=1000,
        # max_token_limit=1000,
        # filter_to_first_n=5,
        # return_scores=True,
        # return_metadata=True,
        # verbose=True,
        # use_bm25=True,
        # use_embedding_database=True,
        # embedding_database_name="moviePlots",
        # embedding_database_property="plotEmbedding",
        # embedding_database_index_name="plotIndex"
    )


# Create a function to call the chain
def get_movie_plot(usr_input):
    """
    Calls the retrieval chain to get the movie plot.

    Args:
        usr_input (str): The question about the movie plot
    """
    neo4jvector = create_neo4j_vector()
    my_retriever = neo4jvector.as_retriever()
    my_prompt = create_prompt()
    my_retrieval_chain = create_my_retrieval_chain(prompt=my_prompt, retriever=my_retriever)

    return my_retrieval_chain.invoke({"input": usr_input})
