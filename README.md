# OCI Postgres Procurement Agent MVP

A lightweight procurement assistant built with **FastAPI + LangChain + OCI GenAI + PostgreSQL/pgvector**.

It helps answer procurement questions using:
- inventory + vendor records stored in Postgres
- policy markdown documents stored in OCI Object Storage

## Features

- Chat API (`/chat`) and streaming chat API (`/chat/stream`)
- Simple web chat UI (`/`)
- Retrieval over:
  - structured procurement data (`inventory_items`, `vendors`) via hybrid search (full-text + vector)
  - procurement policy documents via hybrid chunk search (`policy_chunks` in Postgres)
- Synthetic dataset generator for MVP/demo usage

## Project Structure

```text
app.py                   # FastAPI app + endpoints
core/config.py           # env/config + OCI/Postgres clients
services/agent.py        # LangChain agent setup
services/tools.py        # retrieval/search tools used by the agent
models/schemas.py        # request/response schemas
sql/schema.sql           # DB schema (tables + vector fields)
generate_data.py         # synthetic data generation
ingest_pg.py             # data embedding + ingestion into Postgres
upload_policies.py       # upload policy docs to OCI Object Storage
ui/                      # simple chat UI
```

## Prerequisites

- Python 3.9+
- PostgreSQL with `pgvector` extension
- OCI account and credentials configured (`~/.oci/config` or custom path)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file (do **not** commit it):

```env
# OCI
OCI_CONFIG_FILE=/home/opc/.oci/config
OCI_PROFILE=DEFAULT
OCI_ENDPOINT=<oci_genai_endpoint>
OCI_COMPARTMENT_ID=<ocid1.compartment...>
OCI_MODEL_ID=<chat_model_id>
OCI_EMBED_MODEL_ID=cohere.embed-english-light-v3.0

# Object Storage
OCI_BUCKET=<bucket_name>

# Postgres
PGHOST=<host>
PGPORT=5432
PGDATABASE=<database>
PGUSER=<user>
PGPASSWORD=<password>
PGSSLMODE=require

# Embeddings
EMBED_DIM=384

# Chat session memory
CHAT_MAX_TURNS_PER_SESSION=20
CHAT_MAX_SESSIONS=500
```

Chat history is now persisted to `data/session_history.json`, survives app restarts,
and is bounded by the limits above to avoid unbounded memory growth.

## Data Preparation

1. Generate/refresh synthetic data:

```bash
python generate_data.py
```

2. Ingest inventory/vendors into Postgres with embeddings:

```bash
python ingest_pg.py
```

This ingestion step now also chunks and embeds local policy docs from `data/synthetic/policies/*.md`
into `policy_chunks` for scalable semantic retrieval at query time.

Hybrid ranking combines PostgreSQL full-text relevance (`ts_rank_cd`) with pgvector cosine similarity
for better precision on exact keywords while keeping semantic recall.

3. Upload policy docs to OCI Object Storage:

```bash
python upload_policies.py
```

## Run the App

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Open:
- UI: `http://localhost:8000/`
- Health/basic route: `http://localhost:8000/chat` (GET returns usage help)

## Security Notes

- `.env`, `.env.*`, private key/cert files, virtual env folders, and common secret patterns are ignored via `.gitignore`.
- Never hardcode OCI credentials or DB passwords in source files.

## License

For internal MVP/prototyping use.
