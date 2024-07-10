from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st


def get_chat_llm() -> ChatOpenAI:
    """
    Creates a ChatOpenAI LLM using the provided OpenAI API key and model.

    Returns:
        ChatOpenAI: The ChatOpenAI LLM
    """
    return ChatOpenAI(
        openai_api_key=st.secrets["OPENAI_API_KEY"],
        model=st.secrets["OPENAI_MODEL"]
    )


def create_embeddings() -> OpenAIEmbeddings:
    """
    Creates an OpenAIEmbeddings model using the provided OpenAI API key.

    Returns:
        OpenAIEmbeddings: The OpenAIEmbeddings model
    """
    return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])


# create the llm, and the embeddings
# may now be imported as
# from llm import llm, embeddings
llm = get_chat_llm()
embeddings = create_embeddings()
