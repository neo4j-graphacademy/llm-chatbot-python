from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from llm import llm

# Use the Chains built in the previous lessons
from handlers.vector import kg_qa
from handlers.fewshot import fewshot_cypher_chain

# tag::tools[]
tools = [
    Tool.from_function(
        name="Cypher QA",
        description="Answer information about movies questions using Cypher",
        func = fewshot_cypher_chain,
    ),
    Tool.from_function(
        name="Vector Search Index",
        description="Answer information about movie plots using Vector Search",
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

# tag::agent[]
agent = initialize_agent(
    tools,
    llm,
    memory=memory,
    verbose=True,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    return_intermediate_steps=False
)
# end::agent[]


# tag::generate_response[]
def generate_response(prompt):
    """
    Use the Neo4j Vector Search Index
    to augment the response from the LLM
    """

    response = agent(prompt)

    return response['output']
# end::generate_response[]