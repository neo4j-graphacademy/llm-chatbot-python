from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
# tag::import_prompt[]
from langchain_core.prompts import PromptTemplate
# end::import_prompt[]
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
from utils import get_session_id

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie expert providing information about movies."),
        ("human", "{input}"),
    ]
)

movie_chat = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general movie chat not covered by other tools",
        func=movie_chat.invoke,
    )
]
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

# tag::agent_prompt[]
agent_prompt = PromptTemplate.from_template("""
您是一名儿童中成药专家助手，严格遵守以下规则：
✅ 允许操作：
- 解释药品说明书内容
- 提供年龄分层的剂量建议
- 说明药物相互作用
- 识别常见症状模式
- 引用《中国药典》内容

❌ 禁止操作：
- 诊断疾病或调整处方
- 讨论未批准药品
- 提供非中成药建议
- 处理成人用药问题

安全准则：
1. 遇到发热超过39℃立即建议就医
2. 新生儿（<1月)问题必须转诊
3. 药物过敏史需优先确认

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
Final Answer: [使用中文回答，包含药品名称、适用年龄、关键注意事项]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
# end::agent_prompt[]

# tag::agent[]
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # 增加错误处理
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)
# end::agent[]

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']
