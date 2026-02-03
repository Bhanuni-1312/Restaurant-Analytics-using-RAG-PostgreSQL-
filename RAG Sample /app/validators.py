from __future__ import annotations

import re


def validate_question(question: str, *, max_len: int = 500) -> str:
    """
    Basic but effective input validation.
    - Reject empty
    - Normalize whitespace
    - Limit length to control cost and prevent prompt abuse
    """
    if question is None:
        raise ValueError("Question cannot be null.")

    q = question.strip()
    if not q:
        raise ValueError("Question cannot be empty.")

    # Normalize whitespace (tabs/newlines/multiple spaces -> single space)
    q = re.sub(r"\s+", " ", q)

    if len(q) > max_len:
        raise ValueError(f"Question too long (>{max_len} chars). Please shorten it.")

    return q