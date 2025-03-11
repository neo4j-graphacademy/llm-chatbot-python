import streamlit as st
from llm import llm, embeddings
from pydantic import BaseModel, Field
from graph import graph

# 增强检索模型
class MedicineSearchInput(BaseModel):
    symptoms: str = Field(..., description="症状描述，如咳嗽类型、发热程度")
    age_group: str = Field(..., description="年龄分组:0-1, 1-3, 3-6, 6-12")

# Create the Neo4jVector
from langchain_neo4j import Neo4jVector

neo4jvector = Neo4jVector.from_existing_index(
    embeddings,
    graph=graph,
    index_name="child_medicine_index",       # (3) 自定义索引名称
    node_label="drug",                       # (4) 核心节点类型设为药物
    text_node_property="description",        # (5) 使用药物描述作为文本源
    embedding_node_property="descEmbedding", # (6) 对应嵌入属性名
    retrieval_query="""
RETURN
    node.description AS text,
    score,
    {
        drugName: node.name,
        treatedDiseases: [ (node)<-[:treated_with]-(d:Disease) | d.name ],
        usage: [ (node)-[:has_usage]->(u:Usage) | u.name ],
        targetSymptoms: [
            (d:Disease)-[:has_cardinal_symptom]->(s:Symptom) 
            WHERE d IN [(node)<-[:treated_with]-(d:Disease)]
            | s.name
        ],
        precautions: node.precautions
    } AS metadata
"""
)
# Create the retriever
retriever = neo4jvector.as_retriever(
    search_kwargs={"k": 3, "filter": {"approval_status": "CFDA"}}  # 增加审批状态过滤
)

# Create the prompt
from langchain_core.prompts import ChatPromptTemplate

instructions = """根据以下药品信息回答问题：
- 确保剂量建议匹配用户年龄
- 突出显示禁忌症警告
- 引用来源：{metadata.sources}

若信息不足时：
"建议咨询儿科医师，并参考《{metadata.standard}》相关内容"

上下文：{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", instructions),
    ("human", "患者信息：{age_group}岁儿童，症状：{input}")
])

# Create the chain 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

document_chain = create_stuff_documents_chain(llm, prompt)
medicine_retriever = create_retrieval_chain(
    retriever, 
    document_chain
)


# Create a function to call the chain
def search_child_medicine(input: MedicineSearchInput):
    # 添加年龄过滤
    enhanced_query = f"{input.symptoms} 年龄组：{input.age_group}"
    return medicine_retriever.invoke({
        "input": enhanced_query,
        "age_group": input.age_group
    })

# 药物用法查询函数
def get_drug_usage(drug_name: str, age: int, weight: float):
    cypher = """
    MATCH (d:Drug {name: $name})
    OPTIONAL MATCH (d)-[:has_usage]->(u:Usage)
    RETURN {
        standard_dosage: u.dosage,
        max_daily: u.max_daily,
        administration: u.method,
        adjustment: CASE 
            WHEN $weight > 30 THEN '剂量需按体重调整' 
            ELSE '' 
        END
    } AS usage
    """
    result = graph.run(cypher, {"name": drug_name, "weight": weight}).data()
    return {"usage_info": result[0]['usage'] if result else "无可用数据"}