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
    """Hybrid search inventory and vendors using full-text + pgvector similarity."""
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
                    WITH q AS (
                        SELECT plainto_tsquery('english', %s) AS tsq,
                               %s::vector AS qvec
                    ),
                    ranked AS (
                        SELECT i.item_id,
                               i.item_name,
                               i.category,
                               i.description,
                               i.stock_qty,
                               i.preferred_vendor,
                               i.estimated_price,
                               i.lead_time_days,
                               ts_rank_cd(
                                   to_tsvector(
                                       'english',
                                       coalesce(i.item_name,'') || ' ' || coalesce(i.category,'') || ' ' || coalesce(i.description,'') || ' ' || coalesce(i.preferred_vendor,'')
                                   ),
                                   q.tsq
                               ) AS fts_score,
                               1 - (i.embedding <=> q.qvec) AS vector_score
                        FROM inventory_items i
                        CROSS JOIN q
                    )
                    SELECT item_id,
                           item_name,
                           category,
                           description,
                           stock_qty,
                           preferred_vendor,
                           estimated_price,
                           lead_time_days,
                           fts_score,
                           vector_score,
                           (0.40 * fts_score + 0.60 * vector_score) AS hybrid_score
                    FROM ranked
                    ORDER BY hybrid_score DESC
                    LIMIT 5
                    """,
                    (query, qvec_db),
                )
                items = cur.fetchall()

                cur.execute(
                    """
                    WITH q AS (
                        SELECT plainto_tsquery('english', %s) AS tsq,
                               %s::vector AS qvec
                    ),
                    ranked AS (
                        SELECT v.vendor_id,
                               v.vendor_name,
                               v.categories,
                               v.preferred,
                               v.region,
                               v.sla_days,
                               v.rating,
                               ts_rank_cd(
                                   to_tsvector(
                                       'english',
                                       coalesce(v.vendor_name,'') || ' ' || coalesce(array_to_string(v.categories, ' '), '') || ' ' || coalesce(v.region, '')
                                   ),
                                   q.tsq
                               ) AS fts_score,
                               1 - (v.embedding <=> q.qvec) AS vector_score
                        FROM vendors v
                        CROSS JOIN q
                    )
                    SELECT vendor_id,
                           vendor_name,
                           categories,
                           preferred,
                           region,
                           sla_days,
                           rating,
                           fts_score,
                           vector_score,
                           (0.35 * fts_score + 0.65 * vector_score) AS hybrid_score
                    FROM ranked
                    ORDER BY hybrid_score DESC
                    LIMIT 5
                    """,
                    (query, qvec_db),
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
                    "fts_score": float(r[8] or 0.0),
                    "vector_score": float(r[9] or 0.0),
                    "hybrid_score": float(r[10] or 0.0),
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
                    "fts_score": float(r[7] or 0.0),
                    "vector_score": float(r[8] or 0.0),
                    "hybrid_score": float(r[9] or 0.0),
                }
                for r in vendors
            ],
        }
    except Exception as e:
        return {"error": f"search_procurement_db failed: {e}"}


@tool
def search_procurement_policy(query: str) -> list[dict]:
    """Hybrid search policy chunks using full-text + pgvector semantic similarity."""
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
                    WITH q AS (
                        SELECT plainto_tsquery('english', %s) AS tsq,
                               %s::vector AS qvec
                    ),
                    ranked AS (
                        SELECT COALESCE(p.object_name, p.policy_name) AS object_name,
                               COALESCE(p.chunk_index, 0) AS chunk_index,
                               COALESCE(p.content, p.chunk_text) AS content,
                               ts_rank_cd(
                                   to_tsvector(
                                       'english',
                                       coalesce(p.content, '') || ' ' || coalesce(p.policy_name, '') || ' ' || coalesce(p.object_name, '')
                                   ),
                                   q.tsq
                               ) AS fts_score,
                               1 - (p.embedding <=> q.qvec) AS vector_score
                        FROM policy_chunks p
                        CROSS JOIN q
                    )
                    SELECT object_name,
                           chunk_index,
                           content,
                           fts_score,
                           vector_score,
                           (0.45 * fts_score + 0.55 * vector_score) AS hybrid_score
                    FROM ranked
                    ORDER BY hybrid_score DESC
                    LIMIT 8
                    """,
                    (query, Vector(qvec)),
                )
                rows = cur.fetchall()

        if not rows:
            return object_storage_fallback()

        grouped: dict[str, dict] = {}
        for object_name, chunk_index, content, fts_score, vector_score, hybrid_score in rows:
            bucket = grouped.setdefault(
                object_name,
                {
                    "object_name": object_name,
                    "score": 0.0,
                    "fts_score": 0.0,
                    "vector_score": 0.0,
                    "chunks": [],
                },
            )
            bucket["score"] = max(float(hybrid_score or 0.0), bucket["score"])
            bucket["fts_score"] = max(float(fts_score or 0.0), bucket["fts_score"])
            bucket["vector_score"] = max(float(vector_score or 0.0), bucket["vector_score"])
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
                    "fts_score": round(rec["fts_score"], 4),
                    "vector_score": round(rec["vector_score"], 4),
                    "snippet": snippet,
                    "source": "postgres_hybrid",
                }
            )
        return out
    except Exception as e:
        try:
            fallback = object_storage_fallback()
            return fallback or [{"error": f"search_procurement_policy failed: {e}"}]
        except Exception as fallback_error:
            return [{"error": f"search_procurement_policy failed: {e}; fallback failed: {fallback_error}"}]
