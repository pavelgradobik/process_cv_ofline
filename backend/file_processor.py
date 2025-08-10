import pandas as pd
from bs4 import BeautifulSoup
from backend.config import CSV_PATH
import os

REQUIRED_COLUMNS = {"ID", "Resume_html"}

def html_to_text(html: str) -> str:
    """Extracts main text from HTML using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return ' '.join(text.split())
    except Exception:
        return ""

def load_resumes(csv_path: str = None):
    path = csv_path or CSV_PATH
    if not os.path.exists(path):
        return []

    try:
          df = pd.read_csv(path)
    except Exception:
        return []

    if not REQUIRED_COLUMNS.issubset(df.columns):
        return []

    records = []
    for _, row in df.iterrows():
        html = "" if pd.isna(row.get("Resume_html")) else str(row["Resume_html"])
        text = html_to_text(html)

        rec = {
            "ID": row.get("ID"),
            "Category": row.get("Category", ""),
            "Resume_html": html,
            "Resume_str": text,
        }

        if rec["ID"] is not None and rec["Resume_str"]:
            records.append(rec)

    return records
