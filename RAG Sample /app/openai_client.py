import os
from openai import OpenAI


def get_client() -> OpenAI:
    # Uses OPENAI_API_KEY from environment
    return OpenAI()


def get_chat_model() -> str:
    # Choose a model you listed successfully (gpt-4o-mini, gpt-4, etc.)
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
