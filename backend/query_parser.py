import re
from typing import Dict, List, Optional
from backend.config import ROLE_SYNONYMS

_YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:\+?\s*)?(?:years?|yrs?)", re.IGNORECASE)

def extract_min_years(q: str) -> Optional[float]:
    m = _YEARS_RE.search(q or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def expand_roles(q: str) -> List[str]:
    ql = (q or "").lower()
    out: List[str] = []
    for base, syns in ROLE_SYNONYMS.items():
        if base in ql:
            out.extend(syns)
    seen = set()
    dedup = []
    for s in out:
        if s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup

def build_where(min_years: Optional[float]) -> Dict:
    where: Dict = {}
    if min_years is not None:
        where["years_experience"] = {"$gte": min_years}
    return where

def enrich_query_text(q: str, role_terms: List[str]) -> str:
    parts = [q.strip()] if q else []
    for rt in role_terms:
        if rt.lower() not in (q or "").lower():
            parts.append(rt)
    return "; ".join([p for p in parts if p])