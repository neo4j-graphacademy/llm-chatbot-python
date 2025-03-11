import streamlit as st
from llm import llm
from graph import graph

# Create the Cypher QA chain
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
您是一名专业的Neo4j开发人员,负责将用户关于儿童中成药的提问转换为Cypher查询语句。

请根据以下规则操作：
1. 仅使用schema中定义的关系类型和节点属性
2. 优先查询已批准的儿童用药(approval_status: 'CFDA')
3. 必须包含年龄过滤条件(age_group参数)
4. 不要返回整个节点或embedding属性
5. 剂量查询需关联has_usage关系

特殊处理规则：
- 药品名称需使用标准名称（如"小儿肺热咳喘口服液")
- 年龄参数需转换为分段查询(0-1,1-3,3-6,6-12)
- 症状查询需同时匹配主症(has_cardinal_symptom)和伴随症状(has_concomitant_symptom)

Schema:
{schema}

示例：
问题"3岁儿童咳嗽有痰该用什么药?
查询：
MATCH (d:Disease)-[:has_cardinal_symptom]->(s:Symptom {name:'咳嗽有痰'})
MATCH (d)<-[:treated_with]-(drug:Drug)
WHERE drug.approved_age CONTAINS '3-6'
RETURN drug.name AS medicine, drug.dosage_table AS dosage

当前问题：
{question}

请生成Cypher查询语句:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

from langchain_neo4j import GraphCypherQAChain

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True
)

