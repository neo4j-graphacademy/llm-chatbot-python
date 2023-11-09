# tag::llm[]
import streamlit as st
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model='gpt-4')
# end::llm[]

# tag::embedding[]
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"]
)
# end::embedding[]
