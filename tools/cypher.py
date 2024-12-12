import streamlit as st
from llm import llm
from graph import graph
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from neo4j.exceptions import ClientError, TransientError, DatabaseError

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

def execute_cypher_query(question):
    try:
        response = cypher_qa(question)
        
        # Check if the response contains an error message
        if isinstance(response, str) and "ERROR:" in response:
            return {
                "error": {
                    "type": "invalid_query",
                    "message": response.split("ERROR: ")[1],
                    "suggestion": "Please ask about entities that exist in the database schema."
                }
            }
        
        return {"result": response}
        
    except ClientError as e:
        return {
            "error": {
                "type": "client_error",
                "message": str(e),
                "suggestion": "Please check your query syntax and ensure all referenced entities exist."
            }
        }
        
    except TransientError as e:
        return {
            "error": {
                "type": "transient_error",
                "message": str(e),
                "suggestion": "The database is temporarily unavailable. Please try again in a few moments."
            }
        }
        
    except DatabaseError as e:
        return {
            "error": {
                "type": "database_error",
                "message": str(e),
                "suggestion": "There was an issue with the database. Please contact support if this persists."
            }
        }
        
    except Exception as e:
        return {
            "error": {
                "type": "unknown_error",
                "message": str(e),
                "suggestion": "An unexpected error occurred. Please try a different query."
            }
        }

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,
    cypher_prompt=cypher_prompt,
    return_direct=True
)