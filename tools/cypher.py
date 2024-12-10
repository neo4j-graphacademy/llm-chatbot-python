import streamlit as st
from llm import llm
from graph import graph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about products, orders, and customers in the Northwind database.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Example Cypher Statements:

1. To find products by category:
```
MATCH (p:Product)-[:PART_OF]->(c:Category {{categoryName: "Beverages"}})
RETURN p.productName, p.unitPrice
```

2. To find orders for a customer by customerID (say ALFKI):
```
MATCH (c:Customer {{customerID: "ALFKI"}}-[:PURCHASED]->(o:Order)
RETURN o.orderID, o.orderDate, o.shipAddress
```

3. To find product details with supplier:
```
MATCH (s:Supplier)-[:SUPPLIES]->(p:Product)
WHERE p.productName CONTAINS 'Chai'
RETURN s.companyName, p.productName, p.unitPrice
```

4. To find total number of customers:
```
MATCH (c:Customer)
RETURN COUNT(c) as customerCount
```

Schema:
{schema}

Question:
{question}
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_prompt,
    return_direct=True
)