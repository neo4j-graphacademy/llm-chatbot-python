# tag::importtool[]
from langchain.tools import Tool
# end::importtool[]
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
# tag::importmemory[]
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# end::importmemory[]

from solutions.llm import llm

# Use the Chains built in the previous lessons
from solutions.tools.vector import kg_qa
# from solutions.tools.fewshot import cypher_qa
from solutions.tools.finetuned import cypher_qa

# tag::tools[]
tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
    ),
    Tool.from_function(
        name="Cypher QA",
        description="Provide information about movies questions using Cypher",
        func = cypher_qa,
        return_direct=True
    ),
    Tool.from_function(
        name="Vector Search Index",
        description="Provides information about movie plots using Vector Search",
        func = kg_qa,
        return_direct=True
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

# tag::importprompt[]
from langchain.prompts import PromptTemplate
# end::importprompt[]

# tag::prompt[]
agent_prompt = PromptTemplate.from_template("""
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
# end::prompt[]

# tag::agent[]
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)
# end::agent[]

# tag::generate_response[]
def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent_executor.invoke({"input": prompt})

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
