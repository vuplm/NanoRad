
"""
Text helpers extracted from the original notebooks.
"""
from __future__ import annotations
import re

def extract_pred_text(text: str) -> str:
    """
    Extract the generated assistant text from a chat-formatted string.

    Mirrors the notebooks' behavior: take content after 'assistant\n'
    up to the first '<|im_end|>' token if present.
    """
    if text is None:
        return ""
    # tolerate different line endings
    m = re.search(r"assistant\s*\n(.*)", text, flags=re.DOTALL)
    if not m:
        return text.strip()
    out = m.group(1)
    out = out.split("<|im_end|>")[0]
    return out.strip()
