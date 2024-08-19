import json

from llm import llm
from graph import graph
from tools import cypher_utils
from langchain.prompts.prompt import PromptTemplate
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.chains import LLMChain
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import numpy as np


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
        Chemical concentrations are provided in mg/l.
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
        Do return the name of the requested substance as 'title'.
        Do return all results in JSON format.    
        
        Example Cypher Queries:
        ```
        0. To find measured concentrations of a certain substance
        MATCH (s:Substance)-[r:MEASURED_AT]->(l:Site)
        WHERE s.name = 'Diuron'
        return s.name AS Name, r.median_concentration AS Concentration, l.name AS Site
        ```
        1. To find detected concentrations of a certain substance
        ```
        MATCH (s:Substance)-[r:MEASURED_AT]->(l:Site)
        WHERE s.name='Diuron' AND r.mean_concentration > 0
        RETURN s.name AS Name, r.median_concentration AS Concentration, l.name AS Site
        ```
        
        2. To find the driver importance values of a certain substance
        ```
        MATCH (s:Substance)-[r:IS_DRIVER]->(l:Site)
        WHERE s.name = 'Diuron'
        return s.name AS Name, r.driver_importance AS DriverImportance, l.name AS Site
        ```
        
        Question:
        {question}
        
        Cypher Query:""")


# Find the 10 sites with the highest measured concentration of Diuron and plot these sites and their measured values.
def create_results_plot(results):
    y_axis_name = 'Site'
    x_axis_name_c = 'Concentration'
    x_axis_name_d = 'DriverImportance'
    plot_type = x_axis_name_c
    # results = [{'Name': 'Diuron', 'Concentration': 2.7e-05}, {'Name': 'Diuron', 'Concentration': 1.6e-05},
    #            {'Name': 'Diuron', 'Concentration': 4e-05}, {'Name': 'Diuron', 'Concentration': 1.1e-05},
    #            {'Name': 'Diuron', 'Concentration': 1e-05}, {'Name': 'Diuron', 'Concentration': 1.5e-05},
    #            {'Name': 'Diuron', 'Concentration': 1e-05}, {'Name': 'Diuron', 'Concentration': 2.6e-05},
    #            {'Name': 'Diuron', 'Concentration': 2e-05}, {'Name': 'Diuron', 'Concentration': 2.55e-05}]
    # substance = ""
    if type(results) is list:
        results_df = pd.DataFrame(results)
        columns = results_df.columns
        # idx = next((i for i, value in enumerate(columns.str.contains('Name', case=False)) if value), None)
        # if not idx is None:
        #     substance = results_df[columns[idx]][0]
        # idx = next((i for i, value in enumerate(columns.str.contains(y_axis_name, case=False)) if value), None)
        # if idx is None:
        results_df = results_df.reset_index()
        results_df.rename(columns={'index': 'idx'}, inplace=True)
        y_axis_col_matched = 'idx'
        # else:
        #     y_axis_col_matched = columns[idx]
        idx = next((i for i, value in enumerate(columns.str.contains(x_axis_name_c, case=False)) if value), None)
        if idx is None:
            idx = next((i for i, value in enumerate(columns.str.contains(x_axis_name_d, case=False)) if value), None)
            plot_type = "Driver importance"
        title = plot_type
        x_axis_col_matched = columns[idx]
        # generate plot, check style in: plt.style.available
        plt.style.use('seaborn-v0_8-poster')  # here sth happens
        fig, ax = plt.subplots(figsize=(10, 6))
        # plot points
        scatter = ax.scatter(results_df[x_axis_col_matched], results_df[y_axis_col_matched],
                             s=100, alpha=0.5, c='red', edgecolors='black', linewidth=1.5)
        if plot_type == 'Concentration':
            plt.xscale('log')
        for i in range(len(results_df[x_axis_col_matched])):
            plt.text(results_df[x_axis_col_matched][i], results_df[y_axis_col_matched][i],
                     f'{results_df[x_axis_col_matched][i]}', fontsize=8)

        # add labels and title
        ax.set_ylabel(y_axis_name, fontsize=14)
        ax.set_xlabel(x_axis_name_c if plot_type == x_axis_name_c else x_axis_name_d, fontsize=14)
        ax.set_title(title, fontsize=16)
        # add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        # customize ticks
        # ax.set_xticks(results_df[x_axis_col_matched])
        # ax.set_xticklabels(results_df[x_axis_col_matched], fontsize=12)
        # save plot to file
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png')
        fig_file_name = "figures/plot.png"
        plt.savefig(fig_file_name, format='png')
        plt.close(fig)
        # buf.seek(0)
        # img_bytes = buf.read()
        # image = Image.open(io.BytesIO(img_bytes))
        return fig_file_name
    else:
        return ""


def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str


def invoke_cypher_plot_tool(arg, **kwargs):
    cypher_chain = cypher_utils.create_cypher_qa_chain(prompt_template=create_cypher_prompt_template())
    query_result = cypher_chain.invoke(arg, **kwargs)

    figure_file = create_results_plot(query_result['result'])
    # figure_str = plot_to_base64(figure)

    response_data = {
        # "query_result": query_result,
        "output": query_result,
        "image_url": figure_file,  # Assuming these attributes exist
        # Add other relevant fields as needed
    }

    final_result = json.dumps(response_data)

    return final_result
