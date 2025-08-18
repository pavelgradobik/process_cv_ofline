import os
import sys
import io
from datetime import datetime
import urllib.request
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is first on sys.path (avoid shadowing by site-packages)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cv_backend.config import CSV_PATH as DEFAULT_CSV_PATH
from cv_backend.file_processor import load_resumes_with_stats
from cv_backend.embeddings import OfflineEmbedder
from cv_backend import vector_store as vs
from cv_backend.vector_store import (
    get_client, reset_collection, index_records, query_with_embedding
)
from cv_backend.query_parser import extract_min_years, expand_roles, enrich_query_text


# ---------- helpers ----------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_uploaded_csv(uploaded_file, target_path: str) -> str:
    ensure_dir(os.path.dirname(target_path))
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return target_path

def download_csv(url: str, target_path: str) -> str:
    ensure_dir(os.path.dirname(target_path))
    # use stdlib (no extra deps)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with open(target_path, "wb") as f:
        f.write(data)
    return target_path

def to_results_dataframe(hits: list[dict]) -> pd.DataFrame:
    rows = []
    for h in hits:
        meta = h.get("metadata") or {}
        rows.append({
            "id": h.get("id"),
            "similarity": h.get("similarity"),
            "ID": meta.get("ID"),
            "Title": meta.get("Title"),
            "Skills": meta.get("Skills"),
            "years_experience": meta.get("years_experience"),
            "Category": meta.get("Category"),
            "text": h.get("document"),
        })
    return pd.DataFrame(rows)


# ---------- app layout ----------

st.set_page_config(page_title="Resume Search (Offline)", layout="wide")
st.title("Resume Search (Offline)")

# Show which vector_store file was actually imported (debug/sanity)
st.caption(f"Using vector_store from: `{vs.__file__}` (version: {getattr(vs, '__VERSION__', 'unknown')})")

# Session state
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "records_path" not in st.session_state:
    st.session_state.records_path = DEFAULT_CSV_PATH
if "last_hits" not in st.session_state:
    st.session_state.last_hits = []

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
DOWNLOADS_DIR = os.path.join(DATA_DIR, "downloads")
ensure_dir(DATA_DIR)
ensure_dir(UPLOADS_DIR)
ensure_dir(DOWNLOADS_DIR)

# ---------- Data source controls ----------

with st.expander("Data source", expanded=True):
    source_choice = st.radio(
        "Choose data source",
        ["Use existing CSV", "Upload CSV", "Download CSV from URL"],
        index=0,
        horizontal=True,
    )

    if source_choice == "Use existing CSV":
        st.info(f"Using CSV at: `{st.session_state.records_path}`")
        if st.session_state.records_path != DEFAULT_CSV_PATH:
            st.button("Reset to default CSV", on_click=lambda: st.session_state.update(records_path=DEFAULT_CSV_PATH))

    elif source_choice == "Upload CSV":
        up = st.file_uploader("Upload a CSV of resumes", type=["csv"])
        if up is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_path = os.path.join(UPLOADS_DIR, f"Resume_upload_{ts}.csv")
            save_uploaded_csv(up, target_path)
            st.session_state.records_path = target_path
            st.success(f"Uploaded and set dataset: `{target_path}`")

    elif source_choice == "Download CSV from URL":
        url = st.text_input("CSV URL", placeholder="https://.../Resume.csv")
        col_d1, col_d2 = st.columns([1, 3])
        with col_d1:
            if st.button("Download"):
                if not url.strip():
                    st.error("Please enter a URL.")
                else:
                    try:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        target_path = os.path.join(DOWNLOADS_DIR, f"Resume_download_{ts}.csv")
                        download_csv(url.strip(), target_path)
                        st.session_state.records_path = target_path
                        st.success(f"Downloaded and set dataset: `{target_path}`")
                    except Exception as e:
                        st.exception(e)
        with col_d2:
            st.caption("We use Python's standard library (urllib) to fetch the file; no extra dependencies required.")

# ---------- Load & preview ----------

with st.expander("Load & preview", expanded=True):
    st.write(f"Current dataset: `{st.session_state.records_path}`")
    records, stats = load_resumes_with_stats(st.session_state.records_path)
    if "error" in stats:
        st.error(stats["error"])
    else:
        st.success(
            f"Loaded {stats.get('rows_used', 0)} / {stats.get('total_rows_raw', 0)} usable resumes "
            f"(missing IDs: {stats.get('rows_missing_id', 0)})"
        )
        # tiny preview of first 3 records
        if records:
            sample = []
            for r in records[:3]:
                meta = r["metadata"]
                sample.append({
                    "ID": meta.get("ID"),
                    "Title": meta.get("Title"),
                    "years_experience": meta.get("years_experience"),
                    "Skills": (meta.get("Skills") or "")[:120] + ("..." if meta.get("Skills") and len(meta["Skills"]) > 120 else ""),
                    "text": (r.get("text") or "")[:160] + ("..." if r.get("text") and len(r["text"]) > 160 else ""),
                })
            st.table(pd.DataFrame(sample))

# ---------- Indexing ----------

st.markdown("---")
st.subheader("Index")

col_i1, col_i2 = st.columns([1, 3])
with col_i1:
    rebuild = st.button("Build / Rebuild Index", type="primary")
with col_i2:
    st.caption("Indexes the current dataset into ChromaDB using TF-IDF vectors.")

if rebuild:
    if not records:
        st.error("No records to index.")
    else:
        texts = [r["text"] for r in records]
        embedder = OfflineEmbedder()
        doc_vecs = embedder.fit_transform(texts)

        client = get_client()
        reset_collection(client)
        total_indexed = index_records(records, doc_vecs, client=client, batch=100)

        st.session_state.embedder = embedder
        st.session_state.indexed = True
        st.success(f"Indexed {total_indexed:,} resumes.")

# ---------- Search ----------

st.markdown("---")
st.subheader("Search")

query = st.text_input("Query", placeholder="e.g., python developer with 5 years experience, django")
top_k = st.slider("Top K", 5, 50, 10, 1)
go = st.button("Search")

if go:
    if not st.session_state.indexed or st.session_state.embedder is None:
        st.error("Index not built yet. Click 'Build / Rebuild Index' first.")
    else:
        min_years = extract_min_years(query)
        role_terms = expand_roles(query)
        enriched = enrich_query_text(query, role_terms) or query

        qvec = st.session_state.embedder.transform([enriched])[0]
        hits = query_with_embedding(
            query_embedding=qvec,
            enriched_query_text=enriched,
            min_years=min_years,
            role_terms=role_terms,
            top_k=top_k,
        )

        st.session_state.last_hits = hits or []

        if not hits:
            st.warning("No results.")
        else:
            st.success(f"Found {len(hits)} results.")
            for h in hits:
                meta = h["metadata"] or {}
                st.write(f"**ID**: {meta.get('ID', '—')}  |  **Score**: {h['similarity']}")
                st.write(f"**Title**: {meta.get('Title', '—')}")
                st.write(f"**Years**: {meta.get('years_experience', '—')}  |  **Skills**: {meta.get('Skills', '—')}")
                with st.expander("Text"):
                    st.write(h["document"])
                st.markdown("---")

# ---------- Export results ----------

if st.session_state.last_hits:
    df = to_results_dataframe(st.session_state.last_hits)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results as CSV",
        data=csv_bytes,
        file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )