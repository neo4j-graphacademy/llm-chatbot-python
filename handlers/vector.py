import streamlit as st
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from llm import llm, embeddings
from prompts import GENERAL_SYSTEM_TEMPLATE, GENERAL_USER_TEMPLATE

# tag::vector[]
neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    url=st.secrets["NEO4J_URI"],
    username=st.secrets["NEO4J_USERNAME"],
    password=st.secrets["NEO4J_PASSWORD"],
    index_name=st.secrets["VECTOR_INDEX_NAME"],
    node_label=st.secrets["VECTOR_INDEX_NODE_LABEL"],
    text_node_property=st.secrets["VECTOR_INDEX_TEXT_PROPERTY"],
    embedding_node_property=st.secrets["VECTOR_INDEX_NODE_PROPERTY"],
    retrieval_query="""
RETURN
    node.plot AS text,
    score,
    {
        title: node.title,
        tmdbId: node.tmdbId,
        source: 'https://www.themoviedb.org/movie/'+ node.tmdbId
    } AS metadata
"""
)
# end::vector[]

# tag::retriever[]
retriever = neo4jvector.as_retriever()
# end::retriever[]


# tag::chain[]
messages = [
    SystemMessagePromptTemplate.from_template(GENERAL_SYSTEM_TEMPLATE),
    HumanMessagePromptTemplate.from_template(GENERAL_USER_TEMPLATE),
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

qa_chain = load_qa_with_sources_chain(
    llm,
    chain_type="stuff",
    prompt=qa_prompt,
)
# end::chain[]


# tag::qa[]
kg_qa = RetrievalQAWithSourcesChain(
    combine_documents_chain=qa_chain,
    retriever=retriever,
    return_source_documents=True,
    reduce_k_below_max_tokens=False,
    max_tokens_limit=3375,
    answer_key="answer",
    verbose=True,
)
# end::qa[]

# tag::generate-response[]
def generate_response(prompt):
    """
    Use the Neo4j Vector Search Index
    to augment the response from the LLM
    """

    # Handle the response
    response = kg_qa({"question": prompt})

    return response['answer']
# end::generate-response[]