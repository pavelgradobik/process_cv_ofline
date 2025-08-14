import requests
from backend.config import (
    CORP_API_KEY,
    CORP_EMBED_BASE_URL,
    GENERATIVE_ENGINE_MODEL,
)

def generative_engine_summary(text: str) -> str:
    prompt = (
        "Summarize this candidate's resume in 5 sentences, focusing on their experience, main skills, and professional highlights. Be concise and informative.\n\n"
        f"RESUME:\n{text}\n"
    )
    url = f"{CORP_EMBED_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {CORP_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": GENERATIVE_ENGINE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 200,
    }
    response = requests.post(url, json=data, headers=headers, timeout=60)
    response.raise_for_status()
    out = response.json()
    if "choices" in out and out["choices"]:
        return out["choices"][0]["message"]["content"]
    return "No summary available."
