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
from tools.vector import get_chemical_information
from tools.cypher import invoke_cypher_tool
from tools.wikipedia import wikipedia
from tools.cypher_graph import invoke_cypher_graph_tool
from tools.cypher_plot import invoke_cypher_plot_tool


def create_chemical_chat_chain() -> ChatPromptTemplate:
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an environmental toxicology expert providing information about chemicals that have "
                       "been investigated in European surface waters like rivers and lakes. "
                       "You know about the measured concentrations, the geographical location of the sampling sites, "
                       "and the expected toxicity of the chemicals in algae."
                       "You are also able to evaluate the toxicity of the measured concentration by using the toxic "
                       "unit TU as a reference. "
                       "High values mean higher toxicity. Values above 0.01 may already have an impact on the species,"
                       "and values above 1 are definitely toxic."
                       "You know about the summarized impact in terms of the sum of all TUs (sumTU) of all chemicals "
                       "measured at a sampling site."
                       "You also know about the time point of the sampling as a year and quarter combination."),
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
            description="For general chat not covered by other tools",
            func=general_chat.invoke,
        ),
        # Tool.from_function(
        #     name="Visualize Graph",
        #     description="Provide a graph or network visualisation of chemicals, measured and detected chemical "
        #                 "concentrations in European rivers and lakes.",
        #     func=invoke_cypher_graph_tool,
        # ),
        Tool.from_function(
            name="Plot Cypher Result",
            description="Provide a scientific plot of measured and detected chemical concentrations or chemical "
                        "driver importance values in European rivers and lakes.",
            func=invoke_cypher_plot_tool,
        ),
        Tool.from_function(
            name="Graph DB Search",
            description="Provide details about chemicals and measured and detected chemical concentrations or "
                        "information about chemical driver importance and the time points in the form of year and/or "
                        "quarter information of the measurement time points. "
                        "Information from European surface water bodies like rivers and lakes can be requested.",
            func=invoke_cypher_tool,
        ),
        Tool.from_function(
            name="Wikipedia Search",
            description="For when you need to find general information about a chemical or a sampling site",
            func=wikipedia.invoke,
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
        You are an expert on environmental monitoring of chemicals in European surface waters.
        Be as helpful as possible and return as much information as possible.
        Do not answer any questions that do not relate to chemicals, sampling sites, measured concentrations, 
        toxicities, or species.
        
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
        
        Only use the wikipedia tool for general information.
        For details about measured or detected chemical concentrations in European surface waters like rivers or lakes, 
        use the Chemical measurement information tool.
        Combine the results of different tools to provide as much information to the user as possible.
        For example, use the wikipedia tool before the Chemical measurement information, and include the summary of
        the first paragraph of the wikipedia result as an introduction to the response of the Chemical measurement tool.
        When you have a response including an image_url, keep the image_url in the response.  
        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
        
        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here ([tool_names])]
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
        handle_parsing_errors=True
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
    return response
