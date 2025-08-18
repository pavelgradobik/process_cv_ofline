import os
import sys
import time
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
from cv_backend.vector_store import get_client, reset_collection, index_records, query_with_embedding
from cv_backend.query_parser import extract_min_years, expand_roles, enrich_query_text
from cv_backend.summarizer import pick_best_with_llm

# ⚠️ Import your existing corporate engine module (we won’t assume its class name)
from cv_backend import generative_engine_client as corp

# ---------- helpers ----------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def download_csv(url: str, target_path: str) -> str:
    ensure_dir(os.path.dirname(target_path))
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
            "roles": meta.get("roles"),
            "text": h.get("document"),
        })
    return pd.DataFrame(rows)

def display_hit(h: dict):
    meta = h.get("metadata") or {}
    title = meta.get("Title") or "(no title)"
    st.write(f"**Title**: {title}  |  **ID**: {meta.get('ID', '—')}  |  **Score**: {h.get('similarity')}")
    st.write(f"**Years**: {meta.get('years_experience', '—')}  |  **Skills**: {meta.get('Skills', '—')}")
    with st.expander("Text"):
        st.write(h.get("document"))
    st.markdown("---")

def make_corporate_client():
    """
    Create an instance of your corporate engine client from cv_backend/generative_engine_client.py
    without assuming exact class or factory function name.

    We’ll try, in order:
      - corp.get_client()
      - corp.Client()
      - corp.CorporateClient()
      - corp.CorporateEngine()
      - corp.EngineClient()
      - corp.GenerativeClient()
      - corp.client()  (lowercase)
    If none exist, raise a helpful error.
    """
    # factory function preferred if available
    for func_name in ("get_client", "make_client", "create_client"):
        if hasattr(corp, func_name):
            fn = getattr(corp, func_name)
            return fn()  # type: ignore

    # common class names
    for cls_name in (
        "Client", "CorporateClient", "CorporateEngine", "EngineClient", "GenerativeClient", "client",
    ):
        if hasattr(corp, cls_name):
            cls = getattr(corp, cls_name)
            try:
                return cls()  # type: ignore
            except TypeError:
                # try without args was wrong; try calling with env-based kwargs if present
                try:
                    return cls(None)  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"Could not instantiate corporate engine client ({cls_name}): {e}") from e

    raise RuntimeError(
        "Could not find a client constructor in cv_backend/generative_engine_client.py. "
        "Expose one of: get_client(), Client, CorporateClient, CorporateEngine, EngineClient, GenerativeClient, client."
    )


# ---------- app layout ----------
st.set_page_config(page_title="Resume Search (Offline)", layout="wide")
st.title("Resume Search (Offline)")
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
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
DOWNLOADS_DIR = os.path.join(DATA_DIR, "downloads")
ensure_dir(DATA_DIR); ensure_dir(UPLOADS_DIR); ensure_dir(DOWNLOADS_DIR)

tab_search, tab_llm = st.tabs(["🔎 Search", "🧠 Corporate LLM"])

with tab_search:
    # ---------- Data source ----------
    with st.expander("Data source", expanded=True):
        source_choice = st.radio(
            "Choose data source",
            ["Use existing CSV", "Upload CSV", "Download CSV from URL"],
            index=0, horizontal=True,
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
                with st.status("Saving uploaded file...", expanded=False) as s:
                    ensure_dir(os.path.dirname(target_path))
                    with open(target_path, "wb") as f:
                        f.write(up.getbuffer())
                    s.update(state="complete", label=f"Saved to `{target_path}`")
                st.session_state.records_path = target_path
        elif source_choice == "Download CSV from URL":
            url = st.text_input("CSV URL", placeholder="https://.../Resume.csv")
            if st.button("Download"):
                if not url.strip():
                    st.error("Please enter a URL.")
                else:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    target_path = os.path.join(DOWNLOADS_DIR, f"Resume_download_{ts}.csv")
                    with st.status("Downloading CSV...", expanded=True) as s:
                        try:
                            download_csv(url.strip(), target_path)
                            s.update(state="complete", label=f"Downloaded to `{target_path}`")
                            st.session_state.records_path = target_path
                        except Exception as e:
                            s.update(state="error", label="Download failed")
                            st.exception(e)

    # ---------- Load & preview ----------
    with st.expander("Load & preview", expanded=True):
        with st.status("Loading dataset...", expanded=False) as s:
            records, stats = load_resumes_with_stats(st.session_state.records_path)
            if "error" in stats:
                s.update(state="error", label="Failed to load dataset")
                st.error(stats["error"])
            else:
                s.update(state="complete", label=f"Loaded {stats.get('rows_used', 0)} / {stats.get('total_rows_raw', 0)} resumes")
                if records:
                    df_preview = pd.DataFrame([{
                        "ID": r["metadata"].get("ID"),
                        "Title": r["metadata"].get("Title"),
                        "Years": r["metadata"].get("years_experience"),
                        "Skills": (r["metadata"].get("Skills") or "")[:120] + ("..." if r["metadata"].get("Skills") and len(r["metadata"]["Skills"]) > 120 else ""),
                        "Text": (r.get("text") or "")[:160] + ("..." if r.get("text") and len(r["text"]) > 160 else ""),
                    } for r in records[:3]])
                    st.table(df_preview)

    # ---------- Index ----------
    st.markdown("---")
    st.subheader("Index")
    if st.button("Build / Rebuild Index", type="primary", use_container_width=False):
        if not records:
            st.error("No records to index.")
        else:
            with st.status("Building embeddings...", expanded=True) as s:
                texts = [r["text"] for r in records]
                embedder = OfflineEmbedder()
                doc_vecs = embedder.fit_transform(texts)
                s.update(state="complete", label=f"Built embeddings for {len(texts)} documents")
            with st.status("Indexing to Chroma...", expanded=True) as s:
                client = get_client()
                reset_collection(client)
                total = len(records)
                batch = 100
                prog = st.progress(0)
                indexed = 0
                for start in range(0, total, batch):
                    end = min(start + batch, total)
                    chunk_records = records[start:end]
                    chunk_vecs = doc_vecs[start:end]
                    added = index_records(chunk_records, chunk_vecs, client=client, batch=batch)
                    indexed += added
                    prog.progress(min(1.0, indexed / max(1, total)))
                    time.sleep(0.02)
                s.update(state="complete", label=f"Indexed {indexed:,} resumes")
            st.session_state.embedder = embedder
            st.session_state.indexed = True

    # ---------- Search ----------
    st.markdown("---")
    st.subheader("Search")
    query = st.text_input("Query", placeholder="e.g., hr manager with more than 20 years, recruiter 10+ years")
    TOP_K_FIXED = 8
    if st.button("Search", use_container_width=False):
        if not st.session_state.indexed or st.session_state.embedder is None:
            st.error("Index not built yet. Click 'Build / Rebuild Index' first.")
        else:
            with st.status("Searching...", expanded=False) as s:
                min_years = extract_min_years(query)
                role_terms = expand_roles(query)
                enriched = enrich_query_text(query, role_terms) or query
                qvec = st.session_state.embedder.transform([enriched])[0]
                hits = query_with_embedding(
                    query_embedding=qvec,
                    enriched_query_text=enriched,
                    min_years=min_years,
                    role_terms=role_terms,
                    top_k=TOP_K_FIXED,
                )
                st.session_state.last_hits = hits or []
                st.session_state.last_query = query
                if not hits:
                    s.update(state="complete", label="No results")
                else:
                    s.update(state="complete", label=f"Found {len(hits)} results")
            for h in st.session_state.last_hits:
                display_hit(h)

    # ---------- Export ----------
    if st.session_state.last_hits:
        df = to_results_dataframe(st.session_state.last_hits)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

with tab_llm:
    st.subheader("Corporate LLM — pick the most relevant candidate")
    st.caption("Optional: summarize the current search results and pick the single best match using your corporate engine.")

    disabled = not bool(st.session_state.last_hits)
    if disabled:
        st.info("Run a search first to enable this tab.")

    # controls (kept minimal; your client can ignore kwargs it doesn't use)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05, disabled=disabled)
    consider_n = st.slider("Consider top N results", 3, 20, 8, 1, disabled=disabled)

    if st.button("Ask corporate LLM to pick best", type="primary", disabled=disabled):
        try:
            corp_client = make_corporate_client()
        except Exception as e:
            st.error(str(e))
        else:
            subset = st.session_state.last_hits[:consider_n]
            with st.status("Asking corporate engine...", expanded=False) as s:
                result = pick_best_with_llm(
                    user_query=st.session_state.last_query or "(no query)",
                    hits=subset,
                    client=corp_client,
                    temperature=temperature,
                )
                s.update(state="complete", label="Corporate engine responded")
            best_id = result.get("best_id")
            st.code(result.get("raw", ""), language="json")
            if not best_id:
                st.warning("Corporate engine did not return a best_id.")
            else:
                st.success(f"Chosen candidate: {best_id}")
                # show chosen if in current hits
                by_id = {h.get("id"): h for h in st.session_state.last_hits}
                chosen = by_id.get(best_id)
                if chosen:
                    st.markdown("### Chosen Candidate")
                    display_hit(chosen)
                else:
                    st.info("Chosen candidate is not in the current page of results.")