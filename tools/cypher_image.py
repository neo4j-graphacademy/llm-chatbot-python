import json

from llm import llm
from graph import graph
from tools import cypher_utils
from langchain.prompts.prompt import PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.chains import LLMChain


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
        
        Do return generated plot images of the queried results.
        Do not return text.
        
        Schema:
        {schema}
        
        Context:
        Chemicals are substances.
        The chemical name is stored in the name property of the Substance nodes. 
        The sampling site's name is stored in the name property of the Site nodes.
        Chemical concentrations are stored as mean_concentration and median_concentrations, which are the quarterly 
        summarized concentrations of multiple measurements. 
        Rivers and lakes are water bodies and larger areas around rivers and lake including smaller streams are 
        collected in river basins.
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
        For questions that involve toxicity information use the toxic unit properties 'TU' or 'sumTU' of the relations
        measured_at and summarized_impact_on.
        In case the result contains multiple values, return introductory sentences followed by a list of the values.    
        
        Example Cypher Queries:
        ```
        0. To find measured concentrations of a certain substance
        MATCH (s:Substance)-[r:MEASURED_AT]->(l:Site)
        WHERE s.name = 'Diuron'
        return s.name AS Name, r.median_concentration AS Concentration
        ```
        1. To find detected concentrations of a certain substance
        ```
        MATCH (s:Substance)-[r:MEASURED_AT]->(l:Site)
        WHERE s.name='Diuron' AND r.mean_concentration > 0
        RETURN s.name AS Name, r.median_concentration AS Concentration
        ```
        
        2. To find the driver importance values of a certain substance
        ```
        MATCH (s:Substance)-[r:IS_DRIVER]->(l:Site)
        WHERE s.name = 'Diuron'
        return s.name AS Name, r.driver_importance AS DriverImportance
        ```
        
        Question:
        {question}
        
        Cypher Query:""")


def create_dalle_prompt_template() -> PromptTemplate:
    return PromptTemplate.from_template(
        template="""
            You are a data scientist, aiming at presenting the numbers in your report as scientific plots. 
            You are using the results of a Neo4j Graph Database query to visualise floating point numbers of chemical 
            concentrations and importance of driver chemicals for sampling sites in European water bodies such as rivers and lakes.
            You are able to plot the individual values for the sites.
            You are able to aggregat these numbers and generate summary statistics like average, median, quantiles.
            
               
            Context:
            Chemicals are substances.
            Chemicals are identified by their name and an id, usually the DTXSID or dtxsid.
            Chemical concentrations are measured in mg per liter: mg/l.
            Sampling sites are locations in European water bodies such as rivers and lakes.
            Sampling sites are identified by their name.
            Driver importance values are stored in the driver_importance property of the substance nodes in the graph database.
            Driver importance values range from 0 to 1.
            Concentration values come as mean or median concentrations from the database.
            Concentration values are not negative.
            Measurements have been taken at different quarters in different years.
            
            Instructions:
            Draw scientific plots with a legend and a title.
            If your result contain site names and concentration values or driver importance values, use the y-axis to show the 
            values and the x-axis to show the names of the sites.
            Use the requested Chemical in the plot's title.
            If year and quarter information is provided but no summary or aggregation is requested, generate a plot for each 
            distinct year-quarter pair. 
            If summary or aggregation is requested, compute the summary statistics, and generate a plot showing the requested 
            summary or aggregation, for example a boxplot.
            Use only the first 750 result values.
            
            Generate a detailed prompt shorter than 1000 words to generate an image based on the following description: {image_desc}
            """
    )


def invoke_cypher_plot_tool(arg, **kwargs):
    cypher_chain = cypher_utils.create_cypher_qa_chain(prompt_template=create_cypher_prompt_template())
    query_result = cypher_chain.invoke(arg, **kwargs)

    result_plot = create_results_plot(query_result['result'])

    response_data = {
        # "query_result": query_result,
        "output": ai_message.content,
        "image_url": dalle_response,  # Assuming these attributes exist
        # Add other relevant fields as needed
    }

    final_result = json.dumps(response_data)

    return final_result


def invoke_cypher_plot_tool_with_dalle(arg, **kwargs):
    cypher_chain = cypher_utils.create_cypher_qa_chain(prompt_template=create_cypher_prompt_template())
    query_result = cypher_chain.invoke(arg, **kwargs)

    dalle_prompt_description = create_dalle_prompt_template()
    dalle_chain = dalle_prompt_description | llm
    # dalle_chain = LLMChain(llm=llm, prompt=dalle_prompt_description)
    ai_message = dalle_chain.invoke(arg, **kwargs)
    dalle_response = DallEAPIWrapper().run(ai_message.content)

    response_data = {
        # "query_result": query_result,
        "output": ai_message.content,
        "image_url": dalle_response,  # Assuming these attributes exist
        # Add other relevant fields as needed
    }

    final_result = json.dumps(response_data)

    return final_result
