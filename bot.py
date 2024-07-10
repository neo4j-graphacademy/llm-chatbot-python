import streamlit as st
from utils import write_message
from llm import llm, embeddings
from graph import graph
import agent

# Page Config
st.set_page_config("Ebert", page_icon="🎇")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"},
    ]


# Submit handler
def handle_submit(chat_agent, submitted_message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # # TODO: Replace this with a call to your LLM
        # from time import sleep
        # sleep(1)
        response = agent.generate_response(chat_agent, submitted_message)
        write_message('assistant', response)


movie_chat = agent.create_movie_chat_chain()
tools = agent.create_toolset(general_chat=movie_chat)
chat_agent = agent.create_agent(current_llm=llm, toolset=tools)

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("What is up now?"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(chat_agent=chat_agent, submitted_message=question)
