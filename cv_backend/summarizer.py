import json
from typing import List, Dict, Any, Optional, Callable


def build_selection_prompt(user_query: str, candidates: List[Dict[str, Any]]) -> str:
    lines = [
        "You are an expert technical sourcer.",
        "Given the user query and the list of candidates below, pick the SINGLE most relevant candidate.",
        "Consider role match (including synonyms), years of experience, and key skills.",
        "Respond ONLY with strict JSON like:",
        '{"best_id":"<candidate id>","reason":"<short reason>","notes":"<optional>"}',
        "",
        f"USER QUERY:\n{user_query.strip()}",
        "",
        "CANDIDATES:",
    ]
    for c in candidates:
        text_snip = (c.get("text") or "")[:600]
        lines += [
            f"- id: {c.get('id')}",
            f"  title: {c.get('title')}",
            f"  years_experience: {c.get('years')}",
            f"  skills: {c.get('skills')}",
            f"  text: {text_snip}",
            ""
        ]
    return "\n".join(lines)


def safe_parse_llm_json(s: str) -> Dict[str, Any]:
    if not isinstance(s, str):
        return {"best_id": None, "reason": "LLM returned a non-string payload", "raw": str(s)}
    try:
        obj = json.loads(s.strip())
        if isinstance(obj, dict) and "best_id" in obj:
            return obj
    except Exception:
        pass
    # try to extract a JSON object substring
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start:end + 1])
            if isinstance(obj, dict) and "best_id" in obj:
                return obj
        except Exception:
            pass
    return {"best_id": None, "reason": "LLM returned an unexpected format", "raw": s}


def _call_corporate_llm(client: Any, prompt: str, **kwargs) -> str:
    methods: List[tuple[str, bool]] = [
        ("generate", True), ("generate", False),
        ("complete", True), ("complete", False),
        ("predict",  True), ("predict",  False),
        ("chat",     True), ("chat",     False),
    ]
    for name, use_kwargs in methods:
        if hasattr(client, name):
            fn = getattr(client, name)
            try:
                if use_kwargs:
                    return fn(prompt=prompt, **kwargs)  # type: ignore
                else:
                    return fn(prompt, **kwargs)         # type: ignore
            except TypeError:
                # signature mismatch → try the next variant
                pass

    try:
        return client(prompt=prompt, **kwargs)  # type: ignore
    except TypeError:
        pass
    try:
        return client(prompt, **kwargs)  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Corporate LLM client does not support a known call pattern: {e}") from e


def pick_best_with_llm(
    user_query: str,
    hits: List[Dict[str, Any]],
    client: Any,
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> Dict[str, Any]:
    """
    Use the corporate engine to select the single best candidate from current hits.
    Returns dict with keys: best_id, reason, notes (optional), raw.
    """
    candidates = []
    for h in hits:
        meta = h.get("metadata") or {}
        candidates.append({
            "id": h.get("id"),
            "title": meta.get("Title") or "",
            "years": meta.get("years_experience"),
            "skills": meta.get("Skills") or "",
            "text": h.get("document") or "",
        })
    prompt = build_selection_prompt(user_query, candidates)

    raw = _call_corporate_llm(
        client,
        prompt=prompt,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    parsed = safe_parse_llm_json(raw if isinstance(raw, str) else str(raw))
    parsed["raw"] = raw
    return parsed
