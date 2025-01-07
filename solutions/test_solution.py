import streamlit as st
from streamlit.testing.v1 import AppTest

def test_bot_conversation():

    at = AppTest.from_file(script_path="solutions/bot.py",default_timeout=60).run()
    assert not at.exception, "Bot failed to start"

    question = "What is a good movie about aliens landing on earth?"

    at.chat_input[0].set_value(question).run()

    assert at.chat_message[0].markdown[0].value == "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"
    assert at.chat_message[1].markdown[0].value == question
    assert len(at.chat_message[2].markdown[0].value) > 0 