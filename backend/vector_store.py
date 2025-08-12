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
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )

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
    embed_client,                 # CorporateEmbeddingClient
    client=None,
    batch: int = 1000,
    embed_batch: Optional[int] = None,
):

    client = client or get_client()
    col = get_collection(client)

    ids_all = [str(r["ID"]) for r in records]
    docs_all = [r["Resume_str"] for r in records]
    metas_all = [{"Category": r.get("Category", "")} for r in records]

    vecs_all: List[List[float]] = []
    for s in range(0, len(docs_all), batch):
        e = min(s + batch, len(docs_all))

        vecs_chunk = embed_client.embed_texts(docs_all[s:e])
        vecs_all.extend(vecs_chunk)

        col.add(
            ids=ids_all[s:e],
            documents=docs_all[s:e],
            metadatas=metas_all[s:e],
            embeddings=vecs_chunk,
        )

    return col.count()

def query(
    query_text: str,
    embed_client,
    top_k: int = 10,
    where: Optional[Dict] = None,
    client=None
):
    client = client or get_client()
    col = get_collection(client)

    q_vec = embed_client.embed_query(query_text)

    kwargs = dict(
        query_embeddings=[q_vec],
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
