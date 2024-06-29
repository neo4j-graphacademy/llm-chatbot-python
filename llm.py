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

azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    azure_credential, "https://cognitiveservices.azure.com/.default"
)

llm = AzureChatOpenAI(
    api_version=OPENAI_API_VERSION,
    azure_ad_token_provider=token_provider,
    azure_endpoint=OPENAI_ENDPOINT,
    azure_deployment=MODEL_ID,
    temperature=TEMPERATURE,
    model_kwargs={"seed": SEED},
    streaming=True,
)

embeddings =  AzureOpenAIEmbeddings(
    openai_api_version=OPENAI_API_VERSION,
    azure_deployment=MODEL_ID,
    azure_ad_token_provider=token_provider,
    azure_endpoint=OPENAI_ENDPOINT
)
