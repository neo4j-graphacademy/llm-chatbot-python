from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id

from tools.vector import get_movie_plot
from tools.cypher import cypher_qa

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a company internal service expert chatbot providing information about products, orders, and customers in the Northwind database. "),
        ("human", "{input}"),
    ]
)

movie_chat = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat about the company and its products and services not covered by other tools, Useful for answering generic questions about the company and its products and services",
        func=movie_chat.invoke,
    ), 
    # Tool.from_function(
    #     name="Movie Plot Search",  
    #     description="For when you need to find information about movies based on a plot",
    #     func=get_movie_plot, 
    # ),
    Tool.from_function(
        name="Northwind information",
        description="Provide information about products, orders, and customers in the Northwind database using Cypher",
        func = cypher_qa
    )
]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
You are a store expert providing information about products, orders, and customers in the Northwind database.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to products, orders, or customers.

For each step, you should:
1. Think about whether you need to use a tool
2. Choose the appropriate tool
3. Provide the input to the tool
4. Use the observation to form your final answer

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

def generate_response(user_input, show_intermediate_steps=False):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """
    
    print("Starting generate_response...")  # Debug print
    
    # Configure the agent executor with verbose output
    agent_executor.verbose = True  # Always set to True for debugging
    
    # Call the agent
    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},
    )
    
    print("Raw response:", response)  # Debug print
    
    # Update response handling to be more consistent
    if isinstance(response, dict):
        output = response.get('output', '')
        steps = response.get('intermediate_steps', [])
    else:
        output = str(response)
        steps = []

    if not show_intermediate_steps:
        return output

    intermediate_steps = []
    for step in steps:
        if not isinstance(step, tuple) or len(step) < 1:
            continue
            
        action, observation = step[0], step[1] if len(step) > 1 else None
        step_dict = {}
        
        if hasattr(action, 'log'):
            for line in action.log.split('\n'):
                for key in ['Thought:', 'Action:', 'Action Input:']:
                    if key in line:
                        dict_key = key.rstrip(':')
                        step_dict[dict_key] = line.split(key)[1].strip()
        
        if observation:
            step_dict['Observation'] = str(observation)
            
        if step_dict:
            intermediate_steps.append(step_dict)

    return {
        'output': output,
        'intermediate_steps': intermediate_steps
    }
