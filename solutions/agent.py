from llm import llm
from graph import graph

# tag::import_movie_chat[]
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
# end::import_movie_chat[]

# tag::import_tool[]
from langchain.tools import Tool
# end::import_tool[]

# tag::import_memory[]
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
# end::import_memory[]

# tag::import_get_session_id[]
from utils import get_session_id
# end::import_get_session_id[]


# tag::import_agent[]
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
# end::import_agent[]

# tag::movie_chat[]
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)

movie_chat = chat_prompt | llm | StrOutputParser()
# end::movie_chat[]

# tag::tools[]
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie chat not covered by other tools",
        func=movie_chat.invoke,
    )
]
# end::tools[]

# # Use the Chains built in the previous lessons
# from solutions.tools.vector import kg_qa
# # from solutions.tools.fewshot import cypher_qa
# from solutions.tools.finetuned import cypher_qa

# tag::tools[]
# tools = [
#     Tool.from_function(
#         name="General Chat",
#         description="For general chat not covered by other tools",
#         func=llm.invoke,
#         return_direct=True
#     ),
#     Tool.from_function(
#         name="Cypher QA",
#         description="Provide information about movies questions using Cypher",
#         func = cypher_qa,
#         return_direct=True
#     ),
#     Tool.from_function(
#         name="Vector Search Index",
#         description="Provides information about movie plots using Vector Search",
#         func = kg_qa,
#         return_direct=True
#     )
# ]
# end::tools[]

# tag::get_memory[]
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)
# end::get_memory[]

# tag::agent[]
agent_prompt = hub.pull("hwchase17/react-chat")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)
# end::agent[]

# tag::generate_response[]
def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

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
