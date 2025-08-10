import numpy as np
import os
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

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

def get_collection(client):
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        col = client.create_collection(COLLECTION)
    return col

def reset_collection(client):
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    return client.create_collection(COLLECTION)

def index_records(
    records: List[Dict],
    embeddings,
    client=None,
    batch: int = 1000
):
    client = client or get_client()
    col = get_collection(client)
    N = len(records)
    for start in range(0, N, batch):
        end = min(start + batch, N)
        chunk = records[start:end]
        emb = embeddings[start:end]
        ids = [str(r["ID"]) for r in chunk]
        docs = [r["Resume_str"] for r in chunk]
        metas = [{"Category": r.get("Category", "")} for r in chunk]
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=emb.tolist())
    return col.count()

def query(
    query_text: str,
    query_embedding,
    top_k: int = 10,
    where: Optional[Dict] = None,
    client=None
):
    client = client or get_client()
    col = get_collection(client)

    kwargs = dict(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )
    if where:
        kwargs["where"] = where

    res = col.query(**kwargs)

    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "document": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res.get("distances", [[None]])[0][i],
        })
    return hits

def count(client=None):
    client = client or get_client()
    col = get_collection(client)
    return col.count()
