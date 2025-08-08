import streamlit as st
import os, sys, requests, numpy as np, pandas as pd
from backend.file_processor import load_resumes
from backend.embeddings import OfflineEmbedder

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.set_page_config(page_title="Offline CV Analyzer (with progress)", layout="wide")
st.title("Offline CV Analyzer (with visible stages)")

with st.expander("Upload Resume CSV (local file or by link)", expanded=True):
    uploaded_file = st.file_uploader("Upload Resume CSV", type=["csv"])
    url_upload = st.text_input("Or provide a direct link to a Resume CSV file (e.g. http://.../Resume.csv)")
    upload_btn = st.button("Upload from Link")

    CSV_PATH = os.path.join(PROJECT_ROOT, "data", "uploads", "csv", "Resume.csv")
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    if "resumes_reload" not in st.session_state:
        st.session_state["resumes_reload"] = False

    if uploaded_file is not None:
        with open(CSV_PATH, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Uploaded file saved as {CSV_PATH}")
        st.session_state["resumes_reload"] = True

    elif url_upload and upload_btn:
        try:
            resp = requests.get(url_upload, timeout=30)
            resp.raise_for_status()
            with open(CSV_PATH, "wb") as f:
                f.write(resp.content)
            st.success(f"Downloaded and saved file from {url_upload} → {CSV_PATH}")
            st.session_state["resumes_reload"] = True
        except Exception as e:
            st.error(f"Failed to download: {e}")

def _count_rows(path: str) -> int:
    if not os.path.exists(path): return 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return max(sum(1 for _ in f) - 1, 0)

def load_resumes_with_progress(path: str, chunk_size: int = 2000):
    from backend.file_processor import html_to_text
    if not os.path.exists(path):
        st.error(f"CSV not found at: {path}")
        return []
    total = _count_rows(path)
    prog = st.progress(0, text=f"Reading CSV 0/{total}")
    out, done = [], 0
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        if not {"ID", "Resume_html"}.issubset(chunk.columns):
            st.error("CSV must contain columns: ID, Resume_html (Category optional).")
            return []
        for _, row in chunk.iterrows():
            html = str(row["Resume_html"])
            text = html_to_text(html)
            out.append({"ID": row["ID"], "Category": row.get("Category", ""), "Resume_html": html, "Resume_str": text})
            done += 1
            if done % 200 == 0 or done == total:
                prog.progress((done/total) if total else 1.0, text=f"Reading CSV {done}/{total}")
    prog.empty()
    return out

@st.cache_data(show_spinner=False)
def load_resumes_cached(path: str):
    return load_resumes()

with st.expander("Data loading options"):
    force_fresh = st.checkbox("Force reload with progress", value=False)

if st.session_state["resumes_reload"] or force_fresh:
    with st.status("Loading resumes…", expanded=True) as status:
        st.write("Stage: Reading CSV with progress")
        resumes = load_resumes_with_progress(CSV_PATH)
        status.update(label="Resumes loaded", state="complete", expanded=False)
    st.session_state["resumes_reload"] = False
else:
    with st.spinner("Loading resumes (cached)…"):
        resumes = load_resumes_cached(CSV_PATH)

if not resumes:
    st.warning("No resumes loaded yet.")
    st.stop()

st.caption(f"Loaded {len(resumes):,} resumes.")

texts = [r["Resume_str"] for r in resumes]
with st.status("Building embeddings…", expanded=True) as status:
    st.write("Step 1/2: Fitting TF‑IDF vectorizer")
    embedder = OfflineEmbedder()
    embedder.fit(texts)

    st.write("Step 2/2: Vectorizing all resumes")
    prog = st.progress(0, text=f"Vectorizing 0/{len(texts)}")

    def _update(i, total, _):
        prog.progress(i/total, text=f"Vectorizing {i}/{total}")

    if hasattr(embedder, "embed_batch_with_progress"):
        vectors = embedder.embed_batch_with_progress(texts, update=_update)
    else:
        vectors = embedder.embed_batch(texts)
        prog.progress(1.0, text=f"Vectorizing {len(texts)}/{len(texts)}")

    prog.empty()
    status.update(label="Embeddings ready", state="complete", expanded=False)

tab1, tab2, tab3 = st.tabs(["All Candidates", "Semantic Search", "AI Resume Analysis"])

with tab1:
    st.header("All Candidates")
    st.dataframe(pd.DataFrame(resumes)[["ID", "Category", "Resume_str"]].head(100))

with tab2:
    st.header("Semantic Search")
    query = st.text_input("Query")
    if query:
        q_vec = embedder.embed(query)
        sims = np.dot(vectors, q_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8)
        idxs = np.argsort(sims)[::-1][:10]
        for rank, i in enumerate(idxs, 1):
            r = resumes[i]
            st.subheader(f"{rank}. ID {r['ID']}  (score {sims[i]:.2f})")
            st.write(r["Resume_str"][:800] + " …")

with tab3:
    st.header("AI Resume Analysis")
    st.write("Paste resume text to summarize.")