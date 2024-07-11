from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id
from langchain_core.prompts import PromptTemplate
from tools.vector import get_movie_plot
from tools.cypher import cypher_qa


def create_movie_chat_chain() -> ChatPromptTemplate:
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie expert providing information about movies."),
            ("human", "{input}")
        ]
    )
    return chat_prompt | llm | StrOutputParser()


def create_toolset(general_chat: ChatPromptTemplate) -> [Tool]:
    """
    Creates a set of tools for the movie chat chain.

    Args:
        general_chat (ChatPromptTemplate): The general chat prompt template

    Returns:
        [Tool]: A list of tools
    """
    return [
        Tool.from_function(
            name="General Chat",
            description="For general movie chat not covered by other tools",
            func=general_chat.invoke,
        ),
        Tool.from_function(
            name="Movie Plot Search",
            description="For when you need to find information about movies based on a plot",
            func=get_movie_plot,
        ),
        Tool.from_function(
            name="Movie Information",
            description="Provide information about movies questions using Cypher",
            func=cypher_qa,
        )
        # Add more tools as needed...
    ]


def get_memory(session_id):
    """
    Creates a chat history callback for the Neo4j graph.

    Args:
        session_id (str): The session ID
        neo4j_graph (Neo4jGraph): The current Neo4j graph instance

    Returns:
        Neo4jChatMessageHistory: A chat history object for the given session and graph
    """
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


def get_agent_prompt(standard: bool = True) -> PromptTemplate:
    """
    Returns a prompt template for the movie chat agent.

    Args:
        standard (bool, optional): Whether to use the standard hwchase17/react-chat. Defaults to True.

    Returns:
        PromptTemplate: The movie chat agent prompt template
    """
    if standard:
        return hub.pull("hwchase17/react-chat")
    else:
        return PromptTemplate.from_template("""
        You are a movie expert providing information about movies.
        Be as helpful as possible and return as much information as possible.
        Do not answer any questions that do not relate to movies, actors or directors.
        
        Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.
        
        TOOLS:
        ------
        
        You have access to the following tools:
        
        {tools}
        
        To use a tool, please use the following format:
        
        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ```
        
        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
        
        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```
        
        Begin!
        
        Previous conversation history:
        {chat_history}
        
        New input: {input}
        {agent_scratchpad}
        """)


def create_agent(current_llm, toolset) -> RunnableWithMessageHistory:
    """
    Creates a LangChain agent with the provided LLM, tools, and graph.

    Args:
        current_llm (LLM): The current language model
        toolset (List[Tool]): The set of tools to be used by the agent

    Returns:
        RunnableWithMessageHistory: A runnable agent with the provided LLM, tools, and graph
    """
    agent_prompt = get_agent_prompt(standard=False)
    agent = create_react_agent(llm=current_llm, tools=toolset, prompt=agent_prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=toolset,
        verbose=True,
    )
    return RunnableWithMessageHistory(
        agent_executor,
        get_memory,
        input_messages_key="input",
        history_messages_key="chat_history"
    )


# Create a handler to call the agent
def generate_response(chat_agent, user_input):
    response = chat_agent.invoke(
        {'input': user_input},
        {"configurable": {"session_id": get_session_id()}},
    )
    return response['output']
