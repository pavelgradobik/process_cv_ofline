## Offline CV Analyzer (CSV/XLSX • HTML+STR • ChromaDB)

Fully offline resume processing & semantic search.  
Loads CSV/XLSX, prefers `Resume_str`, falls back to parsing `Resume_html` (BeautifulSoup),  
vectorizes locally (TF‑IDF), persists vectors in **ChromaDB**, and supports fast search.

## How to Run

```bash
# 1) Create venv
python -m venv venv

# 2) Activate
# Windows (PowerShell)
.\venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

# 3) Install deps
pip install -r backend/requirements.txt

# 4) Run UI
streamlit run frontend/app.py


# embeddings stay offline

USE_ENGINE_EMBEDDINGS=false

GENERATIVE_ENGINE_API_KEY=your_key

GENERATIVE_ENGINE_BASE_URL=https://openai.generative.engine.capgemini.com/v1

GENERATIVE_ENGINE_CHAT_MODEL=gpt-4o-mini