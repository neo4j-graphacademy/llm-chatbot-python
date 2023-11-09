from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# The QA Chain for Authoring Cypher
from langchain.chains import GraphCypherQAChain

# The Default prompts used by the QA Chain
from langchain.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT, CYPHER_QA_PROMPT


# tag::finetuned[]

# end::finetuned[]

# tag::fewshot[]
FEWSHOT_CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.
For movie titles that begin with "The", move "the" to the end, For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".

If no context is returned, do not attempt to answer the question.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Schema:
{schema}

Examples:

Find movies and their genres:
MATCH (m:Movie)-[:IN_GENRE]->(g)
WHERE m.title = "Goodfellas"
RETURN m.title AS title, collect(g.name) AS genres

Recommend a movie by actor:
MATCH (subject:Person)-[:ACTED_IN|DIRECTED]->(m)<-[:ACTED_IN|DIRECTED]-(p),
  (p)-[role:ACTED_IN|DIRECTED]->(m2)
WHERE subject.name = "Al Pacino"
RETURN
  m2.title AS recommendation,
  collect([ p.name, type(role) ]) AS peopleInCommon,
  [ (m)-[:IN_GENRE]->(g)<-[:IN_GENRE]-(m2) | g.name ] AS genresInCommon
ORDER BY size(incommon) DESC, size(genresInCommon) DESC LIMIT 2

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Use Neo4j 5 Cypher syntax.  When checking a property is not null, use `IS NOT NULL`.

Question: {question}
"""
# end::fewshot[]