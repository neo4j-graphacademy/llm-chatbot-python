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
from tools.vector import search_child_medicine, get_drug_usage  # 修改工具导入
from tools.cypher import cypher_qa

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
         """您是一名专注于儿童中成药的医疗信息助手，严格遵守以下准则：
1. 仅回答与儿童中成药相关的问题（症状、疾病、用药指导）
2. 用药建议必须基于药品说明书和权威指南
3. 遇到紧急情况立即建议就医
4. 拒绝回答非儿童用药问题"""
         ),
        ("human", "{input}"),
    ]
)

medicine_chat = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="Medical_QA_Boundary",
        description="处理儿童中成药相关问题：症状识别、药物用法、禁忌症。拦截非相关问题",
        func=medicine_chat.invoke,
    ), 
    Tool.from_function(
        name="Symptom_Drug_Search",
        description="根据症状或疾病检索适用药物，需提供：具体症状描述、儿童年龄",
        func=search_child_medicine,  # 连接修改后的检索函数
    ),
    Tool.from_function(
        name="Drug_Usage_Query",
        description="查询药物用法用量，需提供：药品名称、儿童年龄、体重",
        func=get_drug_usage,
        return_direct=True
    ),
    Tool.from_function(
        name="information",
        description="Provide information about questions using Cypher",
        func = cypher_qa
    )
]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

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

def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']