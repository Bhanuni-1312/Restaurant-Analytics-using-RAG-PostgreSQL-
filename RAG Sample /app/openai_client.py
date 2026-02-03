# app/openai_client.py
from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load env variables from .env at import time (safe + simple for small projects)
load_dotenv()


def get_chat_model() -> str:
    """
    Returns the chat model name from env.
    Example in .env:
      OPENAI_CHAT_MODEL=gpt-4o-mini
    """
    return (os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()


def get_client() -> OpenAI:
    """
    Creates and returns an OpenAI client.
    Requires:
      OPENAI_API_KEY=...
    """
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY in environment/.env. "
            "Add it to your .env like: OPENAI_API_KEY=sk-..."
        )

    return OpenAI(api_key=api_key)