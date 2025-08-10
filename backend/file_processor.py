import os
from typing import Dict, List, Tuple
import pandas as pd
from bs4 import BeautifulSoup
from backend.config import CSV_PATH

REQUIRED_ANY_TEXT_COLS = {"Resume_html", "Resume_str"}
REQUIRED_ID_COL = "ID"

def html_to_text(html: str) -> str:
    try:
        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return " ".join(text.split())
    except Exception:
        return ""

def _read_any(path: str) -> pd.DataFrame:
    import pandas as pd
    import os

    ext = os.path.splitext(path.lower())[1]
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")

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
                # Remove low_memory when using engine="python"
                if opts.get("engine") == "python":
                    return pd.read_csv(
                        path,
                        on_bad_lines="skip",
                        encoding=enc,
                        **opts,
                    )
                else:
                    return pd.read_csv(
                        path,
                        low_memory=False,
                        on_bad_lines="skip",
                        encoding=enc,
                        **opts,
                    )
            except Exception as e:
                last_err = e

    try:
        return pd.read_excel(path, engine="openpyxl")
    except Exception:
        raise last_err or RuntimeError("Unable to read file with any strategy")

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {c: c.lower() for c in df.columns}
    df.columns = [c.lower() for c in df.columns]

    alias_map = {
        "resume_html": ["resume_html", "resume html", "resumehtml", "html"],
        "resume_str":  ["resume_str", "resume str", "resumestr", "text"],
        "category":    ["category", "profession", "role", "dept"],
        "id":          ["id", "candidate_id", "candidateid"],
    }

    def find_col(candidates: List[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    id_col = find_col(alias_map["id"])
    html_col = find_col(alias_map["resume_html"])
    str_col = find_col(alias_map["resume_str"])
    cat_col = find_col(alias_map["category"])

    out = pd.DataFrame()
    if id_col: out["ID"] = df[id_col]
    if html_col: out["Resume_html"] = df[html_col]
    if str_col: out["Resume_str"]  = df[str_col]
    if cat_col: out["Category"]    = df[cat_col]
    return out

def _row_to_record(row: pd.Series) -> Dict:
    rid = row.get("ID")
    html = row.get("Resume_html")
    sstr = row.get("Resume_str")
    if pd.isna(sstr) or not str(sstr).strip():
        text = html_to_text(str(html) if not pd.isna(html) else "")
    else:
        text = str(sstr).strip()

    return {
        "ID": rid,
        "Category": ("" if pd.isna(row.get("Category")) else str(row.get("Category"))),
        "Resume_html": "" if pd.isna(html) else str(html),
        "Resume_str": text,
    }

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

    def _non_empty(v) -> bool:
        return not pd.isna(v) and str(v).strip() != ""

    for _, r in df.iterrows():
        if not _non_empty(r.get("ID")):
            missing_id += 1
        html_non_empty += 1 if _non_empty(r.get("Resume_html")) else 0
        str_non_empty += 1 if _non_empty(r.get("Resume_str")) else 0
        if _non_empty(r.get("Resume_str")) or _non_empty(r.get("Resume_html")):
            any_text += 1

    records: List[Dict] = []
    for _, r in df.iterrows():
        if pd.isna(r.get("ID")):
            continue
        has_str_val = _non_empty(r.get("Resume_str"))
        has_html_val = _non_empty(r.get("Resume_html"))
        if not (has_str_val or has_html_val):
            continue
        rec = _row_to_record(r)
        if rec["Resume_str"]:
            records.append(rec)

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