import os
from typing import Dict, List, Tuple
import re
import pandas as pd
from bs4 import BeautifulSoup
from cv_backend.config import CSV_PATH

ROLE_HINTS = [
    r"\b(ui/?ux|ux/?ui)\b",
    r"\b(product|graphic|visual|interaction)\s+designer\b",
    r"\bpython\s+(developer|engineer)\b",
    r"\bsoftware\s+engineer\b",
    r"\bbackend\s+(developer|engineer)\b",
    r"\bfrontend\s+(developer|engineer)\b",
    r"\bqa(\s+engineer)?\b",
]
ROLE_RE = re.compile("|".join(ROLE_HINTS), re.IGNORECASE)
YEARS_ANY_RE = re.compile(r"(?<!\d)(\d{1,2}(?:\.\d+)?)\s*(?:\+?\s*)?(?:years?|yrs?)", re.IGNORECASE)

def _html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())
    except Exception:
        return str(html or "")

def _read_any(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    seps = [None, ",", ";", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(path, encoding=enc, sep=sep, engine="python", on_bad_lines="skip")
            except Exception as e:
                last_err = e
    raise last_err or RuntimeError("Unable to read CSV with any strategy")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    alias = {
        "id":          ["id", "candidate_id", "candidateid"],
        "resume_html": ["resume_html", "resume html", "resumehtml", "html"],
        "resume_str":  ["resume_str", "resume str", "resumestr", "text"],
        "title":       ["title", "current_title", "position"],
        "skills":      ["skills", "skillset", "tech_stack"],
        "category":    ["category", "profession", "role", "dept"],
    }

    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    out = pd.DataFrame()
    if pick(alias["id"]):          out["ID"]          = df[pick(alias["id"])]
    if pick(alias["resume_html"]): out["Resume_html"] = df[pick(alias["resume_html"])]
    if pick(alias["resume_str"]):  out["Resume_str"]  = df[pick(alias["resume_str"])]
    if pick(alias["title"]):       out["Title"]       = df[pick(alias["title"])]
    if pick(alias["skills"]):      out["Skills"]      = df[pick(alias["skills"])]
    if pick(alias["category"]):    out["Category"]    = df[pick(alias["category"])]
    return out

def _extract_roles(text: str) -> List[str]:
    s = set()
    for m in ROLE_RE.finditer(text or ""):
        s.add(m.group(0).lower())
    return sorted(s)

def _extract_years_any(text: str) -> float:
    mx = 0.0
    for m in YEARS_ANY_RE.finditer(text or ""):
        try:
            mx = max(mx, float(m.group(1)))
        except:
            pass
    return mx

def _non_empty(v) -> bool:
    import pandas as pd
    return not pd.isna(v) and str(v).strip() != ""

def load_resumes_with_stats(path: str | None = None) -> Tuple[List[Dict], Dict]:
    p = path or CSV_PATH
    if not os.path.exists(p):
        return [], {"source_path": p, "error": "CSV not found"}

    df_raw = _read_any(p)
    df = _normalize_cols(df_raw)

    total = len(df)
    if total == 0:
        return [], {"source_path": p, "total_rows_raw": 0}

    has_id = "ID" in df.columns
    has_html = "Resume_html" in df.columns
    has_str = "Resume_str" in df.columns

    if not has_id or (not has_html and not has_str):
        return [], {
            "source_path": p,
            "total_rows_raw": total,
            "error": "Missing required columns: need ID and one of Resume_html/Resume_str",
        }

    records: List[Dict] = []
    any_text = 0
    missing_id = 0
    for _, r in df.iterrows():
        cid = str(r["ID"]) if "ID" in df.columns and _non_empty(r["ID"]) else None
        if not cid:
            missing_id += 1
            continue

        raw_html = r.get("Resume_html")
        raw_str  = r.get("Resume_str")
        title    = (r.get("Title") or "").strip() if "Title" in df.columns else ""
        skills   = (r.get("Skills") or "").strip() if "Skills" in df.columns else ""
        category = (r.get("Category") or "").strip() if "Category" in df.columns else ""

        text_html = _html_to_text(raw_html) if _non_empty(raw_html) else ""
        text_str  = str(raw_str).strip() if _non_empty(raw_str) else ""

        full_text = "\n".join([t for t in [title, skills, text_html, text_str] if t])
        if not full_text:
            continue
        any_text += 1

        roles = _extract_roles(" ".join([title, skills, text_html, text_str]))
        years = _extract_years_any(" ".join([title, skills, text_html, text_str]))

        records.append({
            "id": cid,
            "text": full_text,
            "metadata": {
                "ID": cid,
                "Category": category,
                "Title": title,
                "Skills": skills,
                "roles": roles,
                "years_experience": years,
            },
        })

    stats = {
        "source_path": p,
        "total_rows_raw": total,
        "rows_used": len(records),
        "rows_missing_id": missing_id,
    }
    return records, stats

def load_resumes(path: str | None = None) -> List[Dict]:
    items, _ = load_resumes_with_stats(path)
    return items
