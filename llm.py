import streamlit as st

# tag::llm[]
# Create the LLM
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

llm = ChatOpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    model_name=st.secrets["OPENAI_MODEL"]
)
# end::llm[]

# tag::embedding[]
# Create the Embedding model
from langchain_community.embeddings import JinaEmbeddings

# embeddings = JinaEmbeddings(
#     jina_api_key=st.secrets["JINA_API_KEY"],
#     model_name="jina-embeddings-v2-base-en"
# )

embeddings = OpenAIEmbeddings(
    api_key=st.secrets["OPENAI_API_KEY"],
    model="text-embedding-3-small"
)
# end::embedding[]
