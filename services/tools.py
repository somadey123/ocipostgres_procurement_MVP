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
                # Tune IVFFlat recall/speed trade-off per query session.
                # Increase probes for better recall; decrease for lower latency.
                cur.execute("SET LOCAL ivfflat.probes = 10;")
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
    """Search procurement policy content using pgvector semantic similarity over indexed policy chunks."""
    def object_storage_fallback() -> list[dict]:
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
            top.append({"object_name": name, "score": score, "snippet": body[:1200], "source": "object_storage_fallback"})
        return top

    try:
        qvec = embed_text(query)
        with pg_conn() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("SET LOCAL ivfflat.probes = 10;")
                cur.execute(
                    """
                    SELECT COALESCE(object_name, policy_name) AS object_name,
                           COALESCE(chunk_index, 0) AS chunk_index,
                           COALESCE(content, chunk_text) AS content,
                           1 - (embedding <=> %s) AS similarity
                    FROM policy_chunks
                    ORDER BY embedding <=> %s
                    LIMIT 8
                    """,
                    (Vector(qvec), Vector(qvec)),
                )
                rows = cur.fetchall()

        if not rows:
            return object_storage_fallback()

        grouped: dict[str, dict] = {}
        for object_name, chunk_index, content, similarity in rows:
            bucket = grouped.setdefault(
                object_name,
                {"object_name": object_name, "score": 0.0, "chunks": []},
            )
            bucket["score"] = max(float(similarity or 0.0), bucket["score"])
            bucket["chunks"].append((int(chunk_index), content))

        ranked = sorted(grouped.values(), key=lambda x: x["score"], reverse=True)[:3]
        out = []
        for rec in ranked:
            ordered_chunks = [c for _, c in sorted(rec["chunks"], key=lambda x: x[0])]
            snippet = "\n\n".join(ordered_chunks)[:1200]
            out.append(
                {
                    "object_name": rec["object_name"],
                    "score": round(rec["score"], 4),
                    "snippet": snippet,
                    "source": "postgres_vector",
                }
            )
        return out
    except Exception as e:
        try:
            fallback = object_storage_fallback()
            return fallback or [{"error": f"search_procurement_policy failed: {e}"}]
        except Exception as fallback_error:
            return [{"error": f"search_procurement_policy failed: {e}; fallback failed: {fallback_error}"}]
