# tag::importtool[]
from langchain.tools import Tool
# end::importtool[]
from langchain.agents import initialize_agent, AgentType
from langchain.agents import ConversationalChatAgent
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from solutions.llm import llm

# Use the Chains built in the previous lessons
from solutions.tools.vector import kg_qa
# from solutions.tools.finetuned import cypher_qa
from solutions.tools.fewshot import cypher_qa

# tag::tools[]
tools = [
    Tool.from_function(
        name="Cypher QA",
        description="Provide information about movies questions using Cypher",
        func=cypher_qa,
    ),
    Tool.from_function(
        name="Vector Search Index",
        description="Provides information about movie plots using Vector Search",
        func = kg_qa,
    )
]
# end::tools[]


# tag::memory[]
memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)
# end::memory[]

# tag::prompt[]
SYSTEM_MESSAGE = """
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
"""
# end::prompt[]

# tag::agent[]
agent = initialize_agent(
    tools,
    llm,
    memory=memory,
    verbose=True,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    agent_kwargs={"system_message": SYSTEM_MESSAGE}
)
# end::agent[]


# tag::generate_response[]
def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent(prompt)

    return response['output']
# end::generate_response[]


"""

The `generate_response()` method can be called from the `handle_submit()` method in `bot.py`:

# tag::import[]
from agent import generate_response
# end::import[]

# tag::submit[]
# Submit handler
def handle_submit(message):
    # Handle the response
    with st.spinner('Thinking...'):

        response = generate_response(message)
        write_message('assistant', response)
# end::submit[]

"""
