import os
import json
import re
from pathlib import Path

import numpy as np
import oci
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector, Vector

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "synthetic"
POLICY_DIR = DATA_DIR / "policies"

load_dotenv()

EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))
OCI_EMBED_MODEL_ID = os.getenv("OCI_EMBED_MODEL_ID", "cohere.embed-english-light-v3.0")
OCI_ENDPOINT = os.getenv("OCI_ENDPOINT")


def build_genai_client() -> oci.generative_ai_inference.GenerativeAiInferenceClient:
    config_file = os.getenv("OCI_CONFIG_FILE")
    profile = os.getenv("OCI_PROFILE", "DEFAULT")

    # Resolve config path safely and provide a clear fallback.
    if config_file:
        cfg_path = Path(config_file).expanduser()
    else:
        cfg_path = Path.home() / ".oci" / "config"

    if not cfg_path.exists():
        raise FileNotFoundError(
            f"OCI config file not found at: {cfg_path}. "
            "Set OCI_CONFIG_FILE in .env to your real OCI config path (example: /home/opc/.oci/config)."
        )

    config = oci.config.from_file(str(cfg_path), profile)
    return oci.generative_ai_inference.GenerativeAiInferenceClient(
        config=config,
        service_endpoint=OCI_ENDPOINT,
        retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
    )


def chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 120) -> list[str]:
    """
    Sentence-aware chunking with overlap for better semantic retrieval.
    - Keeps chunks under max_chars
    - Adds char-overlap between adjacent chunks to preserve context continuity
    """
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
            # Handle a single very long sentence by hard-splitting.
            for i in range(0, len(sent), max_chars):
                part = sent[i : i + max_chars]
                if part:
                    chunks.append(part)
            current = ""

    if current:
        chunks.append(current)

    return chunks


def embed_text_oci(
    text: str,
    client: oci.generative_ai_inference.GenerativeAiInferenceClient,
    dim: int = EMBED_DIM,
) -> list[float]:
    """
    Create a document embedding by:
    1) chunking text,
    2) embedding each chunk with OCI GenAI,
    3) averaging chunk vectors and L2-normalizing.
    """
    chunks = chunk_text(text)
    details = oci.generative_ai_inference.models.EmbedTextDetails(
        inputs=chunks,
        truncate="END",
        compartment_id=os.environ["OCI_COMPARTMENT_ID"],
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=OCI_EMBED_MODEL_ID
        ),
    )
    response = client.embed_text(details)
    vectors = response.data.embeddings
    if not vectors:
        raise RuntimeError("OCI embedding response was empty")

    arr = np.array(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")
    if arr.shape[1] != dim:
        raise RuntimeError(
            f"Embedding dimension mismatch: model returned {arr.shape[1]}, EMBED_DIM is {dim}. "
            "Update EMBED_DIM and vector column size to match your embedding model."
        )

    pooled = arr.mean(axis=0)
    norm = np.linalg.norm(pooled)
    if norm > 0:
        pooled /= norm
    return pooled.tolist()

def row_text_inventory(row: dict) -> str:
    return f"{row['item_name']} {row['category']} {row['description']} vendor {row.get('preferred_vendor','')}"

def row_text_vendor(row: dict) -> str:
    cats = " ".join(row["categories"])
    return f"{row['vendor_name']} {cats} preferred {row['preferred']} region {row.get('region','')} rating {row.get('rating','')}"

def connect():
    return psycopg.connect(
        host=os.environ["PGHOST"],
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ["PGDATABASE"],
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        sslmode=os.environ.get("PGSSLMODE", "require"),
    )

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_policy_docs(policy_dir: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for file_path in sorted(policy_dir.glob("*.md")):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        docs.append((f"policies/{file_path.name}", text))
    return docs


def inventory_business_key(row: dict) -> tuple:
    return (
        row.get("item_name"),
        row.get("category"),
        row.get("description"),
        row.get("preferred_vendor"),
        row.get("estimated_price"),
        row.get("lead_time_days"),
        row.get("stock_qty"),
    )


def dedupe_inventory_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for row in rows:
        key = inventory_business_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped

def main():
    inventory = dedupe_inventory_rows(list(load_jsonl(DATA_DIR / "inventory.jsonl")))
    vendors = list(load_jsonl(DATA_DIR / "vendors.jsonl"))
    policies = load_policy_docs(POLICY_DIR)
    genai_client = build_genai_client()

    with connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            with open(ROOT / "sql" / "schema.sql", "r", encoding="utf-8") as f:
                cur.execute(f.read())

            cur.execute("DELETE FROM inventory_items;")
            cur.execute("DELETE FROM vendors;")
            cur.execute("DELETE FROM policy_chunks;")

            for row in inventory:
                emb = embed_text_oci(row_text_inventory(row), genai_client)
                cur.execute(
                    """
                    INSERT INTO inventory_items
                    (item_id, item_name, category, description, unit, stock_qty, preferred_vendor, estimated_price, lead_time_days, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        row["item_id"],
                        row["item_name"],
                        row["category"],
                        row["description"],
                        row["unit"],
                        row["stock_qty"],
                        row.get("preferred_vendor"),
                        row.get("estimated_price"),
                        row.get("lead_time_days"),
                        Vector(emb),
                    ),
                )

            for row in vendors:
                emb = embed_text_oci(row_text_vendor(row), genai_client)
                cur.execute(
                    """
                    INSERT INTO vendors
                    (vendor_id, vendor_name, categories, preferred, region, sla_days, rating, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        row["vendor_id"],
                        row["vendor_name"],
                        row["categories"],
                        row["preferred"],
                        row.get("region"),
                        row.get("sla_days"),
                        row.get("rating"),
                        Vector(emb),
                    ),
                )

            for object_name, text in policies:
                chunks = chunk_text(text)
                policy_name = Path(object_name).name
                for chunk_index, chunk in enumerate(chunks):
                    emb = embed_text_oci(chunk, genai_client)
                    cur.execute(
                        """
                        INSERT INTO policy_chunks
                        (policy_name, section_title, chunk_text, keywords, embedding, object_name, chunk_index, content)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (policy_name, None, chunk, [], Vector(emb), object_name, chunk_index, chunk),
                    )

        conn.commit()

    print("Postgres ingestion complete.")

if __name__ == "__main__":
    main()