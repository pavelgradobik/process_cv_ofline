import os
import sys
import requests
import numpy as np
import pandas as pd
import streamlit as st
from backend.file_processor import load_resumes_with_stats, load_resumes
from backend.embeddings import OfflineEmbedder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ChromaDB helpers
from backend.vector_store import (
    get_client,
    reset_collection,
    index_records,
    query as chroma_query,
    count as chroma_count,
)

st.set_page_config(page_title="Offline CV Analyzer (ChromaDB)", layout="wide")
st.title("Offline CV Analyzer (CSV/XLSX • HTML+STR • ChromaDB)")

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "uploads", "csv")
os.makedirs(DATA_DIR, exist_ok=True)
DEFAULT_PATH = os.path.join(DATA_DIR, "Resume.csv")

# Session flags
if "resumes_reload" not in st.session_state:
    st.session_state["resumes_reload"] = False
if "current_source" not in st.session_state:
    st.session_state["current_source"] = DEFAULT_PATH
# reindex control
if "indexed_sig" not in st.session_state:
    st.session_state["indexed_sig"] = None
if "pending_reindex" not in st.session_state:
    st.session_state["pending_reindex"] = False

# ------------- Upload -------------
with st.expander("Upload resumes file (CSV/XLSX) or by link", expanded=True):
    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"])
    url_upload = st.text_input("Or provide a direct link to a CSV/XLSX file")
    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        upload_btn = st.button("Upload from Link")
    with col_u2:
        if st.button("Clear cache"):
            st.cache_data.clear()
            st.success("Streamlit cache cleared.")
    with col_u3:
        if st.button("Reset ChromaDB (manual)"):
            reset_collection(get_client())
            st.session_state["indexed_sig"] = None
            st.success("ChromaDB collection reset.")

    # Save uploaded file (preserve extension) and mark for reindex
    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        target = os.path.join(DATA_DIR, f"Resume{ext or '.csv'}")
        with open(target, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Uploaded → {target}")
        st.session_state["current_source"] = target
        st.session_state["resumes_reload"] = True
        # Reset collection immediately on upload
        reset_collection(get_client())
        st.session_state["indexed_sig"] = None
        st.session_state["pending_reindex"] = True

    elif url_upload and upload_btn:
        try:
            resp = requests.get(url_upload, timeout=60)
            resp.raise_for_status()
            ext = os.path.splitext(url_upload.split("?")[0])[1].lower()
            if ext not in [".csv", ".xlsx", ".xls"]:
                ext = ".csv"
            target = os.path.join(DATA_DIR, f"Resume{ext}")
            with open(target, "wb") as f:
                f.write(resp.content)
            st.success(f"Downloaded → {target}")
            st.session_state["current_source"] = target
            st.session_state["resumes_reload"] = True
            # Reset collection on upload-by-link
            reset_collection(get_client())
            st.session_state["indexed_sig"] = None
            st.session_state["pending_reindex"] = True
        except Exception as e:
            st.error(f"Download failed: {e}")

# -------- Load data (with stats) --------
source_path = st.session_state["current_source"]
st.caption(f"Current source: {source_path}")
try:
    size_bytes = os.path.getsize(source_path)
    st.caption(f"File size: {size_bytes:,} bytes")
except OSError:
    st.caption("File not found (size unavailable).")

with st.expander("Data loading options"):
    force_fresh = st.checkbox("Force reload with progress", value=False)

def load_with_progress(p: str):
    with st.status("Loading resumes…", expanded=True) as s:
        s.write("Stage: Reading file and merging Resume_str / Resume_html")
        records, stats = load_resumes_with_stats(p)
        s.update(label="Resumes loaded", state="complete", expanded=False)
    return records, stats

if st.session_state["resumes_reload"] or force_fresh:
    records, stats = load_with_progress(source_path)
    st.session_state["resumes_reload"] = False
else:
    records = load_resumes(source_path)
    _, stats = load_resumes_with_stats(source_path)

if not records:
    st.info(
        "No usable resumes yet. Upload a CSV/XLSX with **ID** and at least one text column "
        "(**Resume_str** preferred, otherwise **Resume_html** will be parsed)."
    )
    st.caption(
        f"Source: {stats.get('source_path','?')} • "
        f"Total rows: {stats.get('total_rows_raw',0):,} • "
        f"Rows w/ any text: {stats.get('rows_with_any_text',0):,} • "
        f"Rows without text: {stats.get('rows_without_any_text',0):,} • "
        f"Rows missing ID: {stats.get('rows_missing_id',0):,}"
        + (f" • Error: {stats.get('error')}" if stats.get('error') else "")
    )
    st.stop()

st.caption(
    f"Source: {stats.get('source_path','?')} • "
    f"Total rows (raw): {stats.get('total_rows_raw',0):,} • "
    f"Rows with any text: {stats.get('rows_with_any_text',0):,} • "
    f"Rows without text: {stats.get('rows_without_any_text',0):,} • "
    f"Rows missing ID: {stats.get('rows_missing_id',0):,} • "
    f"Used (valid) resumes: {stats.get('rows_used',0):,} • "
    f"non-empty HTML: {stats.get('html_non_empty',0):,} • non-empty STR: {stats.get('str_non_empty',0):,}"
)

# -------- Build embeddings (in-memory) --------
texts = [r["Resume_str"] for r in records]
with st.status("Building embeddings…", expanded=True) as status:
    status.write("Step 1/2: Fitting TF‑IDF vectorizer")
    embedder = OfflineEmbedder()
    embedder.fit(texts)

    status.write(f"Step 2/2: Vectorizing {len(texts):,} resumes")
    prog = st.progress(0, text=f"Vectorizing 0/{len(texts)}")

    if hasattr(embedder, "embed_batch_with_progress"):
        def _update(i, total, _):
            prog.progress(i / total, text=f"Vectorizing {i}/{total}")
        vectors = embedder.embed_batch_with_progress(texts, update=_update)
    else:
        vectors = embedder.embed_batch(texts)
        prog.progress(1.0, text=f"Vectorizing {len(texts)}/{len(texts)}")

    prog.empty()
    status.update(label="Embeddings ready", state="complete", expanded=False)

st.caption(f"In‑memory vectors: {vectors.shape[0]:,} × {vectors.shape[1]}")

# -------- Index into ChromaDB only when needed --------
def file_signature(path: str, records_len: int) -> tuple:
    try:
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
    except OSError:
        mtime = 0.0
        size = 0
    return (os.path.abspath(path), round(mtime, 3), size, int(records_len))

current_sig = file_signature(source_path, len(records))
need_index = False

# Reindex if:
#  - we flagged pending_reindex (just uploaded), or
#  - collection is empty, or
#  - file signature changed since last index
try:
    col_count = chroma_count(get_client())
except Exception:
    col_count = 0

if st.session_state["pending_reindex"] or col_count == 0 or st.session_state["indexed_sig"] != current_sig:
    need_index = True

if need_index:
    with st.status("Indexing into ChromaDB…", expanded=True) as s:
        client = get_client()
        # If we didn't already reset on upload, ensure types/dims match by resetting now
        if not st.session_state["pending_reindex"] and col_count > 0:
            reset_collection(client)
        total_indexed = index_records(records, vectors, client=client, batch=1000)
        st.session_state["indexed_sig"] = current_sig
        st.session_state["pending_reindex"] = False
        s.write(f"Indexed {total_indexed:,} resumes.")
        s.update(label="ChromaDB ready", state="complete", expanded=False)

st.caption(f"Chroma collection size: {chroma_count(get_client()):,}")

tab1, tab2 = st.tabs(["All Candidates", "Semantic Search (ChromaDB)"])

with tab1:
    st.header("All Candidates")
    df = pd.DataFrame(records)
    st.dataframe(df[["ID", "Category", "Resume_str"]].head(200))
    candidate_id = st.text_input("View candidate by ID")
    if candidate_id:
        try:
            cid = int(candidate_id)
        except Exception:
            cid = candidate_id
        cand = next((r for r in records if r["ID"] == cid), None)
        if cand:
            st.write(f"**ID:** {cand['ID']}")
            st.write(f"**Category:** {cand['Category']}")
            st.write("**Extracted/Provided Resume Text:**")
            st.write(cand["Resume_str"])
            with st.expander("Show Raw Resume HTML"):
                st.markdown(cand["Resume_html"], unsafe_allow_html=True)
        else:
            st.warning("Candidate not found.")

with tab2:
    st.header("Semantic Search (ChromaDB)")
    query_text = st.text_input("Query (e.g., 'Senior Python developer 10 years')")
    if query_text:
        q_vec = embedder.embed(query_text)
        # Fixed top_k = 10, no category filter, and where=None (do NOT pass {})
        hits = chroma_query(query_text, q_vec, top_k=10, where=None)
        if not hits:
            st.warning("No results.")
        else:
            for rank, h in enumerate(hits, 1):
                st.subheader(f"{rank}. ID {h['id']}")
                meta = h.get("metadata") or {}
                if meta.get("Category"):
                    st.caption(f"Category: {meta['Category']}")
                doc = h.get("document") or ""
                st.write(doc[:1200] + ("…" if len(doc) > 1200 else ""))