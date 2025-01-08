import pytest

def test_secrets():
    import streamlit as st

    def check_secret(key):
        assert len(st.secrets[key]) > 0, f"{key} not found in secrets.toml"
    
    try:
        check_secret("OPENAI_API_KEY")
        check_secret("OPENAI_MODEL")
        check_secret("NEO4J_URI")
        check_secret("NEO4J_USERNAME")
        check_secret("NEO4J_PASSWORD")

    except FileNotFoundError:
        assert False, "secrets.toml file not found"

def test_vector():
    try:
        from tools.vector import get_movie_plot
        assert get_movie_plot("Aliens land on earth") is not None

        vector_exists = True

    except ValueError:
        assert False, "The moviePlots index does not exist. Run the Cypher script - https://raw.githubusercontent.com/neo4j-graphacademy/courses/refs/heads/main/asciidoc/courses/llm-chatbot-python/modules/3-tools/lessons/1-vector-tool/reset.cypher"

    assert True

def test_bot_conversation():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(script_path="solutions/bot.py",default_timeout=60).run()
    assert not at.exception, "Bot failed to start"

    question = "What is a good movie about aliens landing on earth?"

    at.chat_input[0].set_value(question).run()

    assert at.chat_message[0].markdown[0].value == "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"
    assert at.chat_message[1].markdown[0].value == question
    assert len(at.chat_message[2].markdown[0].value) > 0, "No response from the bot"