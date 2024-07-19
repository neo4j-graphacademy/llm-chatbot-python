from llm import llm
from graph import graph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate


def create_cypher_qa_chain(prompt_template: PromptTemplate) -> GraphCypherQAChain:
    """
    Creates a GraphCypherQAChain using the provided prompt template and Neo4j graph.

    Args:
        prompt_template (PromptTemplate): The prompt template for Cypher queries
    """
    chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        verbose=True,
        cypher_prompt=prompt_template,
        return_intermediate_steps=True,
    )
    chain.return_direct = True
    return chain
