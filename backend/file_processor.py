import os
from typing import Dict, List, Tuple
import re
import pandas as pd
from bs4 import BeautifulSoup
from backend.config import CSV_PATH

REQUIRED_ANY_TEXT_COLS = {"Resume_html", "Resume_str"}
REQUIRED_ID_COL = "ID"

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

def html_to_text(html: str) -> str:
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
    try_orders = [
        dict(sep=None, engine="python"),
        dict(sep=",", engine="python"),
        dict(sep=";", engine="python"),
        dict(sep="\t", engine="python"),
        dict(sep="|", engine="python"),
    ]
    last_err = None
    for enc in encodings:
        for opts in try_orders:
            try:
                if opts.get("engine") == "python":
                    return pd.read_csv(path, on_bad_lines="skip", encoding=enc, **opts)
                else:
                    return pd.read_csv(path, low_memory=False, on_bad_lines="skip", encoding=enc, **opts)
            except Exception as e:
                last_err = e
    raise last_err or RuntimeError("Unable to read file with any strategy")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    alias_map = {
        "resume_html": ["resume_html", "resume html", "resumehtml", "html"],
        "resume_str":  ["resume_str", "resume str", "resumestr", "text"],
        "category":    ["category", "profession", "role", "dept"],
        "id":          ["id", "candidate_id", "candidateid"],
        "title":       ["title", "current_title", "position"],
        "skills":      ["skills", "skillset", "tech_stack"],
    }

    def find_col(candidates: List[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    id_col   = find_col(alias_map["id"])
    html_col = find_col(alias_map["resume_html"])
    str_col  = find_col(alias_map["resume_str"])
    cat_col  = find_col(alias_map["category"])
    title_col= find_col(alias_map["title"])
    skills_col= find_col(alias_map["skills"])

    out = pd.DataFrame()
    if id_col:    out["ID"]          = df[id_col]
    if html_col:  out["Resume_html"] = df[html_col]
    if str_col:   out["Resume_str"]  = df[str_col]
    if cat_col:   out["Category"]    = df[cat_col]
    if title_col: out["Title"]       = df[title_col]
    if skills_col:out["Skills"]      = df[skills_col]
    return out

def _extract_roles(text: str) -> List[str]:
    roles = set()
    for m in ROLE_RE.finditer(text or ""):
        roles.add(m.group(0).lower())
    return sorted(roles)

def _extract_years_any(text: str) -> float:
    max_years = 0.0
    for m in YEARS_ANY_RE.finditer(text or ""):
        try:
            max_years = max(max_years, float(m.group(1)))
        except Exception:
            pass
    return max_years

def load_resumes_with_stats(path: str | None = None) -> Tuple[List[Dict], Dict]:
    p = path or CSV_PATH
    if not os.path.exists(p):
        return [], {
            "total_rows_raw": 0,
            "rows_with_any_text": 0,
            "rows_without_any_text": 0,
            "rows_missing_id": 0,
            "rows_used": 0,
            "source_path": p,
        }

    try:
        df_raw = _read_any(p)
    except Exception as e:
        return [], {
            "total_rows_raw": 0,
            "rows_with_any_text": 0,
            "rows_without_any_text": 0,
            "rows_missing_id": 0,
            "rows_used": 0,
            "source_path": p,
            "error": f"Failed to read file: {type(e).__name__}: {e}"
        }

    df = _normalize_cols(df_raw)

    total = len(df)
    if total == 0:
        return [], {
            "total_rows_raw": 0,
            "rows_with_any_text": 0,
            "rows_without_any_text": 0,
            "rows_missing_id": 0,
            "rows_used": 0,
            "source_path": p,
        }

    has_id = "ID" in df.columns
    has_html = "Resume_html" in df.columns
    has_str = "Resume_str" in df.columns

    if not has_id or (not has_html and not has_str):
        return [], {
            "total_rows_raw": total,
            "rows_with_any_text": 0,
            "rows_without_any_text": total,
            "rows_missing_id": total if not has_id else 0,
            "rows_used": 0,
            "source_path": p,
            "error": "Missing required columns (need ID and one of Resume_html/Resume_str)",
        }

    html_non_empty = 0
    str_non_empty = 0
    any_text = 0
    missing_id = 0
    records: List[Dict] = []

    def _non_empty(v) -> bool:
        return not pd.isna(v) and str(v).strip() != ""

    for _, r in df.iterrows():
        cid = str(r["ID"]) if "ID" in r and _non_empty(r["ID"]) else None
        if not cid:
            missing_id += 1
            continue

        raw_html = r.get("Resume_html")
        raw_str  = r.get("Resume_str")
        title    = str(r.get("Title") or "").strip()
        skills   = str(r.get("Skills") or "").strip()
        category = str(r.get("Category") or "").strip()

        text_html = html_to_text(raw_html) if _non_empty(raw_html) else ""
        text_str  = str(raw_str).strip() if _non_empty(raw_str) else ""
        if text_html: html_non_empty += 1
        if text_str:  str_non_empty += 1

        if not (text_html or text_str):
            continue

        full_text = "\n".join([t for t in [title, skills, text_html, text_str] if t])
        any_text += 1

        roles = _extract_roles(" ".join([title, skills, text_html, text_str]))
        years = _extract_years_any(" ".join([title, skills, text_html, text_str]))

        metadata = {
            "ID": cid,
            "Category": category,
            "Title": title,
            "Skills": skills,
            "roles": roles,  # normalized hints
            "years_experience": years,  # numeric for filtering
        }

        records.append({
            "id": cid,
            "text": full_text,
            "metadata": metadata,
        })

    stats = {
        "total_rows_raw": int(total),
        "rows_with_any_text": int(any_text),
        "rows_without_any_text": int(total - any_text),
        "rows_missing_id": int(missing_id),
        "rows_used": int(len(records)),
        "source_path": p,
        "html_non_empty": int(html_non_empty),
        "str_non_empty": int(str_non_empty),
    }
    return records, stats

def load_resumes(path: str | None = None) -> List[Dict]:
    records, _ = load_resumes_with_stats(path)
    return records
