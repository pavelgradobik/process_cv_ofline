from typing import List, Dict
from backend.config import CHAT_MODEL
from backend.engine_client import EngineClient

SYSTEM_PROMPT = (
    "You are an assistant that answers hiring queries using only the provided resume snippets. "
    "Summarize and rank the most relevant candidates. Be concise and factual. "
    "If information is missing, say so."
)

def build_context_snippets(hits: List[Dict], max_chars: int = 8000) -> str:
    parts, length = [], 0
    for h in hits:
        doc = h.get("document") or ""
        cid = h.get("id")
        cat = (h.get("metadata") or {}).get("Category", "")
        header = f"[ID: {cid} | Category: {cat}]"
        snippet = f"{header}\n{doc}\n---\n"
        if length + len(snippet) > max_chars:
            break
        parts.append(snippet)
        length += len(snippet)
    return "".join(parts)

def answer_query(query: str, hits: List[Dict]) -> str:
    client = EngineClient()
    context = build_context_snippets(hits)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Use only the following resume snippets:\n\n{context}\n\n"
                "Return:\n- A short answer to the query\n- A bullet list of top 5 candidate IDs with 1-line reasons\n"
            ),
        },
    ]
    return client.chat(CHAT_MODEL, messages, temperature=0.2)