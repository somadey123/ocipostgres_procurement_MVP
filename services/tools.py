import os
import re

import numpy as np
import oci
from langchain_core.tools import tool
from pgvector.psycopg import Vector, register_vector

from core.config import embed_dim, oci_embed_client, oci_embed_model_id, oci_object_storage_client, pg_conn


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 120) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return [""]
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if not sent:
            continue
        proposal = f"{current} {sent}".strip() if current else sent
        if len(proposal) <= max_chars:
            current = proposal
            continue
        if current:
            chunks.append(current)
            overlap = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = f"{overlap} {sent}".strip()
        else:
            for i in range(0, len(sent), max_chars):
                part = sent[i : i + max_chars]
                if part:
                    chunks.append(part)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def embed_text(text: str) -> list[float]:
    """Use same OCI embedding strategy as ingestion: chunk -> embed -> mean pool -> normalize."""
    dim = embed_dim()
    client = oci_embed_client()
    chunks = chunk_text(text)
    details = oci.generative_ai_inference.models.EmbedTextDetails(
        inputs=chunks,
        truncate="END",
        compartment_id=os.environ["OCI_COMPARTMENT_ID"],
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=oci_embed_model_id()
        ),
    )
    response = client.embed_text(details)
    vectors = response.data.embeddings
    if not vectors:
        raise RuntimeError("OCI embedding response was empty")
    arr = np.array(vectors, dtype=np.float32)
    if arr.shape[1] != dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: model returned {arr.shape[1]}, EMBED_DIM is {dim}."
        )
    pooled = arr.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled /= norm
    return pooled.tolist()


@tool
def search_procurement_db(query: str) -> dict:
    """Search inventory and preferred vendors in PostgreSQL using pgvector similarity."""
    try:
        qvec = embed_text(query)
        with pg_conn() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                qvec_db = Vector(qvec)

                cur.execute(
                    """
                    SELECT item_id, item_name, category, description, stock_qty, preferred_vendor, estimated_price, lead_time_days
                    FROM inventory_items
                    ORDER BY embedding <=> %s
                    LIMIT 5
                    """,
                    (qvec_db,),
                )
                items = cur.fetchall()

                cur.execute(
                    """
                    SELECT vendor_id, vendor_name, categories, preferred, region, sla_days, rating
                    FROM vendors
                    ORDER BY embedding <=> %s
                    LIMIT 5
                    """,
                    (qvec_db,),
                )
                vendors = cur.fetchall()

        return {
            "inventory": [
                {
                    "item_id": r[0],
                    "item_name": r[1],
                    "category": r[2],
                    "description": r[3],
                    "stock_qty": r[4],
                    "preferred_vendor": r[5],
                    "estimated_price": float(r[6]) if r[6] is not None else None,
                    "lead_time_days": r[7],
                }
                for r in items
            ],
            "vendors": [
                {
                    "vendor_id": r[0],
                    "vendor_name": r[1],
                    "categories": r[2],
                    "preferred": r[3],
                    "region": r[4],
                    "sla_days": r[5],
                    "rating": float(r[6]) if r[6] is not None else None,
                }
                for r in vendors
            ],
        }
    except Exception as e:
        return {"error": f"search_procurement_db failed: {e}"}


@tool
def search_procurement_policy(query: str) -> list[dict]:
    """Search procurement policy documents stored in OCI Object Storage."""
    try:
        client = oci_object_storage_client()
        namespace = client.get_namespace().data
        bucket = os.environ["OCI_BUCKET"]

        objs = client.list_objects(namespace, bucket, prefix="policies/").data.objects
        terms = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 2]

        scored = []
        for obj in objs:
            body = (
                client.get_object(namespace, bucket, obj.name)
                .data.content.decode("utf-8", errors="ignore")
            )
            score = sum(body.lower().count(t) for t in terms)
            scored.append((score, obj.name, body))

        scored.sort(reverse=True, key=lambda x: x[0])
        top = []
        for score, name, body in scored[:3]:
            top.append({"object_name": name, "score": score, "snippet": body[:1200]})
        return top
    except Exception as e:
        return [{"error": f"search_procurement_policy failed: {e}"}]
