from functools import cache

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from settings import (
    LLM_MODEL_ID,
    OPENAI_API_VERSION,
    OPENAI_ENDPOINT,
    TEMPERATURE,
    SEED,
    EMBEDDING_MODEL_ID
)

azure_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    azure_credential, "https://cognitiveservices.azure.com/.default"
)

llm = AzureChatOpenAI(
    api_version=OPENAI_API_VERSION,
    azure_ad_token_provider=token_provider,
    azure_endpoint=OPENAI_ENDPOINT,
    azure_deployment=LLM_MODEL_ID,
    temperature=TEMPERATURE,
    model_kwargs={"seed": SEED},
    streaming=True,
)

embeddings =  AzureOpenAIEmbeddings(
        azure_ad_token_provider=token_provider,
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=EMBEDDING_MODEL_ID
)

if __name__ == "__main__":
    from langchain_core.messages.human import HumanMessage

    result = llm.invoke([HumanMessage(content="what is 2+2")])
    print(result)