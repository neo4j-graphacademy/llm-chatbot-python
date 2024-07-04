import os
from dotenv import load_dotenv

# load_dotenv(".envrc")
load_dotenv(".env")

OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT", "")
LLM_MODEL_ID = "gpt-4o"
OPENAI_API_VERSION = "2024-02-01"
TEMPERATURE = 0.1
SEED = 191

EMBEDDING_MODEL_ID = "ada-002"