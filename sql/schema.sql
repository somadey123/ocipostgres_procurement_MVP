CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS inventory_items (
    item_id TEXT PRIMARY KEY,
    item_name TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL,
    unit TEXT NOT NULL,
    stock_qty INT NOT NULL,
    preferred_vendor TEXT,
    estimated_price NUMERIC(12,2),
    lead_time_days INT,
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS vendors (
    vendor_id TEXT PRIMARY KEY,
    vendor_name TEXT NOT NULL,
    categories TEXT[] NOT NULL,
    preferred BOOLEAN NOT NULL DEFAULT FALSE,
    region TEXT,
    sla_days INT,
    rating NUMERIC(3,2),
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS policy_chunks (
    chunk_id BIGSERIAL PRIMARY KEY,
    -- Legacy columns kept for backward compatibility with older deployments
    policy_name TEXT NOT NULL,
    section_title TEXT,
    chunk_text TEXT NOT NULL,
    keywords TEXT[] NOT NULL DEFAULT '{}',

    -- Canonical columns for current semantic retrieval path
    object_name TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    UNIQUE (object_name, chunk_index)
);

-- Backward-compatible migration for environments where policy_chunks already exists
-- with an older column layout.
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS policy_name TEXT;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS section_title TEXT;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS chunk_text TEXT;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS keywords TEXT[] DEFAULT '{}';
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS object_name TEXT;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS chunk_index INT;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS content TEXT;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS embedding VECTOR(384);

-- Fill canonical columns from legacy ones if needed.
UPDATE policy_chunks
SET object_name = COALESCE(object_name, policy_name),
    chunk_index = COALESCE(chunk_index, 0),
    content = COALESCE(content, chunk_text),
    policy_name = COALESCE(policy_name, object_name),
    chunk_text = COALESCE(chunk_text, content),
    keywords = COALESCE(keywords, '{}');

-- Enforce expected constraints after backfill.
ALTER TABLE policy_chunks ALTER COLUMN policy_name SET NOT NULL;
ALTER TABLE policy_chunks ALTER COLUMN chunk_text SET NOT NULL;
ALTER TABLE policy_chunks ALTER COLUMN keywords SET NOT NULL;
ALTER TABLE policy_chunks ALTER COLUMN object_name SET NOT NULL;
ALTER TABLE policy_chunks ALTER COLUMN chunk_index SET NOT NULL;
ALTER TABLE policy_chunks ALTER COLUMN content SET NOT NULL;

-- Vector ANN indexes for faster semantic similarity search (ORDER BY embedding <=> query LIMIT K)
-- Using cosine distance because query path uses the <=> operator.
CREATE INDEX IF NOT EXISTS idx_inventory_items_embedding_cosine_ivfflat
ON inventory_items
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_vendors_embedding_cosine_ivfflat
ON vendors
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_embedding_cosine_ivfflat
ON policy_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
