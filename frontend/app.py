import os
import sys
import streamlit as st
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cv_backend.config import CSV_PATH
from cv_backend.file_processor import load_resumes_with_stats
from cv_backend.embeddings import OfflineEmbedder
from cv_backend import vector_store as vs
from cv_backend.vector_store import (
    get_client, reset_collection, index_records, query_with_embedding
)
from cv_backend.query_parser import extract_min_years, expand_roles, enrich_query_text

st.set_page_config(page_title="Resume Search (Offline)", layout="wide")
st.title("Resume Search (Offline)")

# Show the exact module path being used
st.caption(f"Using vector_store from: `{vs.__file__}` (version: {getattr(vs, '__VERSION__', 'unknown')})")

# ====== Data Loading ======
with st.expander("Data", expanded=True):
    st.write(f"Source CSV: `{CSV_PATH}`")
    records, stats = load_resumes_with_stats(CSV_PATH)
    if "error" in stats:
        st.error(stats["error"])
    else:
        st.success(f"Loaded {stats.get('rows_used', 0)} / {stats.get('total_rows_raw', 0)} usable resumes")

# ====== Build Embeddings & Index ======
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False

col_a, col_b = st.columns([1,1])
with col_a:
    if st.button("Build / Rebuild Index", type="primary"):
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
with col_b:
    st.write("")  # spacer

st.markdown("---")

# ====== Search ======
query = st.text_input("Search query", placeholder="e.g., python developer with 5 years experience, django")
top_k = st.slider("Top K", 5, 50, 10, 1)

if st.button("Search"):
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

        if not hits:
            st.warning("No results.")
        else:
            for h in hits:
                meta = h["metadata"] or {}
                st.write(f"**ID**: {meta.get('ID', '—')}  |  **Score**: {h['similarity']}")
                st.write(f"**Title**: {meta.get('Title', '—')}")
                st.write(f"**Years**: {meta.get('years_experience', '—')}  |  **Skills**: {meta.get('Skills', '—')}")
                with st.expander("Text"):
                    st.write(h["document"])
                st.markdown("---")
