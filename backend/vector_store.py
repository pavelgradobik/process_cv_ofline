import os
from typing import List, Dict, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from backend.query_parser import extract_min_years, expand_roles, build_where, enrich_query_text

DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "chromadb"
)
COLLECTION = "cv_embeddings"

def get_client(persist_dir: Optional[str] = None):
    persist_dir = persist_dir or DEFAULT_DIR
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
    return client

def get_collection(client=None):
    client = client or get_client()
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        return client.create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

def reset_collection(client=None):
    client = client or get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    return client.create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

def index_records(
    client,
    ids: List[str],
    documents: List[str],
    metadatas: List[Dict],
    vectors: np.ndarray
):
    col = get_collection(client)
    vecs = vectors.tolist()
    col.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=vecs)

def _role_boost_score(meta: Dict, doc: str, role_terms: List[str]) -> float:
    hay = f"{meta.get('Title') or ''} {meta.get('Skills') or ''} {doc or ''}".lower()
    score = 0.0
    for t in role_terms:
        if t.lower() in hay:
            score += 0.05
    return min(score, 0.5)  # cap

def query(
    query_text: str,
    top_k: int = 10,
) -> List[Dict]:
    client = get_client()
    col = get_collection(client)

    min_years = extract_min_years(query_text)
    role_terms = expand_roles(query_text)
    where = build_where(min_years)
    enriched = enrich_query_text(query_text, role_terms) or query_text

    res = col.query(
        query_texts=[enriched],
        n_results=max(top_k, 20),
        where=where,
        include=["documents", "metadatas", "distances", "embeddings", "ids"]
    )

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0] or [0.0] * len(ids)

    items: List[Tuple[str, float, Dict, str]] = []
    for i in range(len(ids)):
        _id = ids[i]
        doc = docs[i]
        meta = metas[i] or {}
        base = 1.0 - float(dists[i])
        boost = _role_boost_score(meta, doc, role_terms)
        items.append((_id, base + boost, meta, doc))

    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:top_k]

    hits = []
    for _id, score, meta, doc in items:
        hits.append({
            "id": _id,
            "document": doc,
            "metadata": meta,
            "similarity": round(score, 4),
        })
    return hits

def search_by_text(query_text: str, top_k: int = 10) -> List[Dict]:
    return query(query_text, top_k=top_k)

def count(client=None):
    client = client or get_client()
    col = get_collection(client)
    return col.count()