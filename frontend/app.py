import streamlit as st
import pandas as pd
from backend.file_processor import load_resumes
from backend.embeddings import OfflineEmbedder
from backend.summary_service import get_resume_summary
import numpy as np
import os, sys
import tempfile
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

st.title("Offline CV Analyzer with Capgemini Generative Engine")

# File upload tab
with st.expander("Upload Resume CSV (local file or by link)", expanded=True):
    uploaded_file = st.file_uploader("Upload Resume CSV", type=["csv"])
    url_upload = st.text_input("Or provide a direct link to a Resume CSV file (e.g. http://.../Resume.csv)")
    upload_btn = st.button("Upload from Link")
    
    csv_save_path = "data/uploads/csv/Resume.csv"
    file_changed = False

    if uploaded_file is not None:
        # Save uploaded file
        with open(csv_save_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Uploaded file saved as {csv_save_path}")
        file_changed = True

    elif url_upload and upload_btn:
        try:
            resp = requests.get(url_upload)
            resp.raise_for_status()
            with open(csv_save_path, "wb") as f:
                f.write(resp.content)
            st.success(f"Downloaded and saved file from {url_upload} as {csv_save_path}")
            file_changed = True
        except Exception as e:
            st.error(f"Failed to download: {e}")

    if file_changed:
        st.session_state["resumes_reload"] = True

if "resumes_reload" not in st.session_state:
    st.session_state["resumes_reload"] = False

@st.cache_data(show_spinner=False)
def get_all_resumes():
    from backend.file_processor import load_resumes
    return load_resumes()

if st.session_state["resumes_reload"]:
    from backend.file_processor import load_resumes
    resumes = load_resumes()
    st.session_state["resumes_reload"] = False
else:
    resumes = get_all_resumes()

texts = [r["Resume_str"] for r in resumes]

st.set_page_config(page_title="CV Analyzer (w/ Capgemini AI)", layout="wide")

st.title("Offline CV Analyzer with Capgemini Generative Engine")

resumes = load_resumes()
texts = [r["Resume_str"] for r in resumes]

embedder = OfflineEmbedder()
embedder.fit(texts)
vectors = embedder.embed_batch(texts)

def semantic_search(query, top_k=5):
    q_vec = embedder.embed(query)
    sims = np.dot(vectors, q_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec) + 1e-8)
    idxs = np.argsort(sims)[::-1][:top_k]
    return [resumes[i] for i in idxs], [sims[i] for i in idxs]

tab1, tab2, tab3 = st.tabs(["All Candidates", "Semantic Search", "AI Resume Analysis"])

with tab1:
    st.header("All Candidates")
    df = pd.DataFrame(resumes)
    st.dataframe(df[["ID", "Category", "Resume_str"]].head(100))
    candidate_id = st.text_input("View candidate by ID")
    if candidate_id:
        cand = next((r for r in resumes if str(r["ID"]) == candidate_id), None)
        if cand:
            st.write(f"**ID:** {cand['ID']}")
            st.write(f"**Category:** {cand['Category']}")
            st.write("**Extracted Resume Text:**")
            st.write(cand["Resume_str"])
            with st.expander("Show Raw Resume HTML"):
                st.markdown(cand["Resume_html"], unsafe_allow_html=True)
            st.write("**AI/Extractive Summary:**")
            with st.spinner("Generating summary..."):
                st.info(get_resume_summary(cand["Resume_str"]))
        else:
            st.error("Candidate not found")

with tab2:
    st.header("Semantic Search")
    query = st.text_input("Enter your search query (e.g. 'Senior Python developer with 10 years experience')")
    if query:
        results, scores = semantic_search(query, top_k=10)
        for i, (res, score) in enumerate(zip(results, scores), 1):
            st.subheader(f"Rank #{i} (Score: {score:.2f})")
            st.write(f"**ID:** {res['ID']}")
            st.write(f"**Category:** {res['Category']}")
            st.write("**Resume:**")
            st.write(res["Resume_str"][:700] + " ...")
            with st.expander("Show AI/Extractive Summary"):
                st.info(get_resume_summary(res["Resume_str"]))

with tab3:
    st.header("AI Resume Analysis")
    st.write("Paste a candidate's resume text below to get an AI/extractive summary.")
    user_text = st.text_area("Resume text", height=300)
    if st.button("Analyze Resume"):
        if user_text.strip():
            with st.spinner("Generating summary..."):
                st.info(get_resume_summary(user_text))
        else:
            st.warning("Paste some resume text above.")

st.caption("All local/offline. Summaries use Capgemini Generative Engine if available, otherwise local summarizer")
