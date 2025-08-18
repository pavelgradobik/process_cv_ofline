import os
import json
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import chromadb
from chromadb.config import Settings

from cv_backend.query_parser import build_where

__VERSION__ = "vector_store/1.4.0"

DEFAULT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "chromadb"
)
COLLECTION = "cv_embeddings"

def get_client(persist_dir: Optional[str] = None):
    persist_dir = persist_dir or DEFAULT_DIR
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

def get_collection(client=None):
    client = client or get_client()
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        return client.create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

def reset_collection(client=None):
    client = client or get_client()
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    return client.create_collection(name=COLLECTION, metadata={"hnsw:space": "cosine"})

# ----------------- helpers -----------------

def _as_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _to_primitive(v: Any) -> Any:
    # convert numpy scalars to native
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    # everything else (lists/dicts/objects) → JSON string
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

def _sanitize_metadatas(metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for m in metas:
        if not isinstance(m, dict):
            cleaned.append({"_meta": _to_primitive(m)})
            continue
        cleaned.append({k: _to_primitive(v) for k, v in m.items()})
    return cleaned

def _index_batches(col, ids: List[str], docs: List[str], metas: List[Dict], vecs: np.ndarray, batch: int) -> int:
    n = len(ids)
    total = 0
    batch = max(1, int(batch))
    metas = _sanitize_metadatas(metas)  # ← sanitize once

    for start in range(0, n, batch):
        end = min(start + batch, n)
        col.add(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metas[start:end],
            embeddings=_as_list(vecs[start:end]),
        )
        total += (end - start)
    return total

# --------------- public API ----------------

def index_records(*args, **kwargs) -> int:
    """
    Backward-compatible indexer.

    (A) NEW STYLE:
        index_records(records, vectors, client=None, batch=100)

    (B) OLD STYLE:
        index_records(client, ids, documents, metadatas, vectors, batch=100)
    """
    # NEW style: first arg is list[dict]
    if len(args) >= 2 and isinstance(args[0], list):
        records: List[Dict[str, Any]] = args[0]
        vectors: np.ndarray = args[1]
        client = kwargs.get("client", None)
        batch = kwargs.get("batch", 100)

        if vectors is None:
            raise ValueError("vectors must not be None")
        if len(records) != len(vectors):
            raise ValueError(f"records ({len(records)}) and vectors ({len(vectors)}) length mismatch")

        col = get_collection(client or get_client())
        ids = [r["id"] for r in records]
        docs = [r["text"] for r in records]
        metas = [r.get("metadata", {}) for r in records]
        return _index_batches(col, ids, docs, metas, vectors, batch)

    # OLD style:
    if len(args) < 5:
        raise TypeError(
            "index_records expected either (records, vectors, ...) or (client, ids, documents, metadatas, vectors, ...)"
        )
    client = args[0]
    ids = args[1]
    docs = args[2]
    metas = args[3]
    vectors = args[4]
    batch = kwargs.get("batch", 100)

    col = get_collection(client)
    if not (len(ids) == len(docs) == len(metas) == len(vectors)):
        raise ValueError("ids, documents, metadatas, and vectors must have the same length")
    return _index_batches(col, ids, docs, metas, vectors, batch)

def _role_boost_score(meta: Dict, doc: str, role_terms: List[str]) -> float:
    hay = f"{meta.get('Title') or ''} {meta.get('Skills') or ''} {doc or ''}".lower()
    score = 0.0
    for t in role_terms:
        if t.lower() in hay:
            score += 0.05
    return min(score, 0.5)

def query_with_embedding(
    query_embedding: np.ndarray,
    enriched_query_text: str,
    min_years: Optional[float],
    role_terms: List[str],
    top_k: int = 8,
) -> List[Dict]:
    col = get_collection()
    where = build_where(min_years)
    # Chroma expects None when no filter; {} will raise
    if not where:
        where = None

    res = col.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=max(top_k, 20),
        where=where,
        include=["documents", "metadatas", "distances"],  # never include "ids"
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

    return [
        {
            "id": _id,
            "document": doc,
            "metadata": meta,
            "similarity": round(score, 4),
        } for _id, score, meta, doc in items
    ]