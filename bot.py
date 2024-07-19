import _io
import re
import streamlit as st
from utils import write_message
from llm import llm, embeddings
from graph import graph
import agent
import requests
from io import BytesIO
from PIL import Image
import base64

# Page Config
st.set_page_config(page_title="EcoToxFred", page_icon="figures/assistant.png")

# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi, I'm EcoToxFred!  How can I help you?",
         "avatar": "figures/simple_avatar.png"},
    ]


# Submit handler
def handle_submit(chat_agent, submitted_message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Generating response ...'):
        response = agent.generate_response(chat_agent, submitted_message)
        find_str = 'figures/plot.png'
        pattern = re.compile(r'\b\w*' + re.escape(find_str) + r'\w*\b')
        matches = pattern.findall(response['output'])
        if len(matches) > 0:
            for match in matches:
                response['output'] = response['output'].replace(match, "")
            write_message('assistant', response['output'])
            st.image("figures/plot.png", caption="Image generated with matplotlib from graph db cypher query result.")
        else:
            write_message('assistant', response['output'])


my_chat = agent.create_chemical_chat_chain()
tools = agent.create_toolset(general_chat=my_chat)
chat_agent = agent.create_agent(current_llm=llm, toolset=tools)

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if question := st.chat_input("What do you want to know?"):
    # Display user message in chat message container
    write_message('user', question)

    # Generate a response
    handle_submit(chat_agent=chat_agent, submitted_message=question)
