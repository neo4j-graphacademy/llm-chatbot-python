from llm import llm
from graph import graph
from tools import cypher_utils
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate


def create_cypher_prompt_template():
    """
    Creates a prompt template for Cypher queries.

    Returns:
        PromptTemplate: The Cypher prompt template
    """
    return PromptTemplate.from_template("""
        You are an expert Neo4j Developer translating user questions into Cypher to answer questions about chemicals, 
        and their measured concentrations in European surface waters like rivers and lakes. 
        Convert the user's question based on the schema.
        
        Use only the provided relationship types and properties in the schema.
        Do not use any other relationship types or properties that are not provided.
        
        Do not return entire nodes or embedding properties.
        
        Schema:
        {schema}
        
        Context:
        Chemicals are substances.
        The chemical name is stored in the name property of the Substance nodes. 
        The sampling site's name is stored in the name property of the Site nodes.
        Chemical concentrations are stored as mean_concentration and median_concentrations, which are the quarterly 
        summarized concentrations of multiple measurements. 
        Rivers and lakes are water bodies. 
        Water body names are stored in the water_body property.
        Larger areas around rivers and lake including smaller streams are collected in river basins.
        The DTXSID referes to the CompTox Dashboard ID of the U.S. EPA.
        The verb detected in the context of chemical monitoring referes to a measured concentration above 0.
        
        Instructions:
        Ignore water_body and country in case you are only asked about finding information on a certain chemical.
        If you cannot find the requested chemical name, ask the user to provide the Comptox Dashboard ID of the 
        requested chemical which is the DTXSID.
        For questions that involve time or the interrogative 'when' refer to the node and relation properties year 
        and quarter.
        If you cannot find the requested river name in water_body search in river_basin and vice versa.
        For questions that involve geographic or location information or the interrogative 'where' search in the 
        properties of the Site nodes.
        For questions that involve European rivers, lakes or water bodies search in the properties of the Site nodes.
        For questions that involve toxicity information use the toxic unit properties 'TU' or 'sumTU' of the relations
        measured_at and summarized_impact_on.
        In case the result contains multiple values, return introductory sentences followed by a list of the values.
    
        
        Example Cypher Queries:
        0. To find information about a certain substance
        ```
        MATCH (s:Substance) 
        WHERE s.name = 'Diuron'
        return s.DTXSID as DTXSID, s.name as Name
        ```
        1. To find sites where a certain substance has been measured
        ```
        MATCH (s:Substance)-[r:MEASURED_AT]->(l:Site)
        WHERE s.name='Diuron'
        RETURN s.name as ChemicalName, s.DTXSID as DTXSID, r.concentration_value as Concentration, 
        r.concentration_unit as ConcentrationUnit, r.year as Year, r.quarter as Quarter, l.name as SiteName, 
        l.country as Country, l.river_basin as RiverBasin, l.water_body as WaterBody
        ```
        
        2. To find sites where a certain substance has been measured with a mean concentration above a threshold:
        ```
        MATCH (s:Substance)-[r:MEASURED_AT]->(l:Site)
        WHERE s.name='Diuron' AND r.mean_concentration > 0.001
        RETURN s.name as ChemicalName, s.DTXSID as DTXSID, r.concentration_value as Concentration, 
        r.concentration_unit as ConcentrationUnit, r.year as Year, r.quarter as Quarter, l.name as SiteName, 
        l.country as Country, l.river_basin as RiverBasin, l.water_body as WaterBody
        ```
        
        3. To find all chemicals measured in a certain river:
        ```
        MATCH (c:Substance)-[r:MEASURED_AT]->(s:Site)
        WHERE s.water_body = 'seine' AND r.mean_concentration > 0.001
        RETURN c.DTXSID AS DTXSID, c.name AS Name
        ```
        
        4. To find all substances measured in France
        ```
        MATCH (c:Substance)-[r:MEASURED_AT]->(s:Site)
        WHERE s.country = 'France'
        RETURN DISTINCT c.DTXSID AS DTXSID, c.name AS Name
        ```
        
        5. To find the 10 most frequent driver chemicals above a driver importance of 0.6 
        ```
        MATCH (s:Substance)-[r:IS_DRIVER]->(l:Site)
        WHERE r.driver_importance > 0.6
        RETURN s.name AS substance, COUNT(r) AS frequency
        ORDER BY frequency DESC
        LIMIT 10
        ```
        
        6. To find the substances with highest driver importance, provide the first 10
        ```
        MATCH (s:Substance)-[r:IS_DRIVER]->(l:Site) 
        WHERE r.driver_importance>0.8 
        RETURN DISTINCT s.name, s.DTXSID, r.driver_importance 
        ORDER BY r.driver_importance
        LIMIT 10
        ```
        
        Question:
        {question}
        
        Cypher Query:""")


# def create_cypher_qa_chain(prompt_template: PromptTemplate) -> GraphCypherQAChain:
#     """
#     Creates a GraphCypherQAChain using the provided prompt template and Neo4j graph.
#
#     Args:
#         prompt_template (PromptTemplate): The prompt template for Cypher queries
#     """
#     return GraphCypherQAChain.from_llm(
#         llm,
#         graph=graph,
#         verbose=True,
#         cypher_prompt=prompt_template,
#         return_embedding=False,  # Don't return embedding properties for better performance
#         validate_cypher=True,
#         return_intermediate_steps=True,
#     )


def invoke_cypher_tool(arg, **kwargs):
    chain = cypher_utils.create_cypher_qa_chain(prompt_template=create_cypher_prompt_template())
    query_result = chain.invoke(arg, **kwargs)
    return str(query_result)


# cypher_qa = create_cypher_qa_chain(prompt_template=create_cypher_prompt_template())
