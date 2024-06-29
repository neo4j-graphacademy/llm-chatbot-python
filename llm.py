from functools import cache

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from settings import (
    MODEL_ID,
    OPENAI_API_VERSION,
    OPENAI_ENDPOINT,
    TEMPERATURE,
    SEED,
)

@cache
def init_llm(
    model_id=MODEL_ID,
    api_version=OPENAI_API_VERSION,
    temperature=TEMPERATURE,
    seed=SEED,
):
    azure_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        azure_credential, "https://cognitiveservices.azure.com/.default"
    )

    return AzureChatOpenAI(
        api_version=api_version,
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=model_id,
        temperature=temperature,
        model_kwargs={"seed": seed},
        streaming=True,
    )

@cache
def init_embeddings(
    model_id=MODEL_ID,
    api_version=OPENAI_API_VERSION
):
    azure_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        azure_credential, "https://cognitiveservices.azure.com/.default"
    )

    return AzureOpenAIEmbeddings(
        api_version=api_version,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        azure_deployment=model_id
    )

if __name__ == "__main__":
    from langchain_core.messages.human import HumanMessage
    embeddings = init_embeddings()
    print("embeddings done")


    llm = init_llm()
    result = llm.invoke([HumanMessage(content="hello world")])
    print(result)