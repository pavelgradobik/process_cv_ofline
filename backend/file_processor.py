import pandas as pd
from bs4 import BeautifulSoup
from backend.config import CSV_PATH

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

def load_resumes():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["ID", "Resume_html"])
    records = []
    for idx, row in df.iterrows():
        html = str(row["Resume_html"])
        text = html_to_text(html)
        records.append({
            "ID": row["ID"],
            "Category": row.get("Category", ""),
            "Resume_html": html,
            "Resume_str": text,  # Always update with parsed text!
        })
    return records
