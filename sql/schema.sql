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
    search_vector tsvector,
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
    search_vector tsvector,
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
    search_vector tsvector,
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
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS search_vector tsvector;
ALTER TABLE policy_chunks ADD COLUMN IF NOT EXISTS embedding VECTOR(384);

ALTER TABLE inventory_items ADD COLUMN IF NOT EXISTS search_vector tsvector;
ALTER TABLE vendors ADD COLUMN IF NOT EXISTS search_vector tsvector;

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

CREATE OR REPLACE FUNCTION inventory_items_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector(
        'english',
        coalesce(NEW.item_name, '') || ' ' || coalesce(NEW.category, '') || ' ' || coalesce(NEW.description, '') || ' ' || coalesce(NEW.preferred_vendor, '')
    );
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION vendors_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector(
        'english',
        coalesce(NEW.vendor_name, '') || ' ' || coalesce(array_to_string(NEW.categories, ' '), '') || ' ' || coalesce(NEW.region, '')
    );
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION policy_chunks_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector(
        'english',
        coalesce(NEW.content, '') || ' ' || coalesce(NEW.policy_name, '') || ' ' || coalesce(NEW.object_name, '')
    );
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_inventory_items_search_vector ON inventory_items;
CREATE TRIGGER trg_inventory_items_search_vector
BEFORE INSERT OR UPDATE ON inventory_items
FOR EACH ROW EXECUTE FUNCTION inventory_items_search_vector_update();

DROP TRIGGER IF EXISTS trg_vendors_search_vector ON vendors;
CREATE TRIGGER trg_vendors_search_vector
BEFORE INSERT OR UPDATE ON vendors
FOR EACH ROW EXECUTE FUNCTION vendors_search_vector_update();

DROP TRIGGER IF EXISTS trg_policy_chunks_search_vector ON policy_chunks;
CREATE TRIGGER trg_policy_chunks_search_vector
BEFORE INSERT OR UPDATE ON policy_chunks
FOR EACH ROW EXECUTE FUNCTION policy_chunks_search_vector_update();

UPDATE inventory_items
SET search_vector = to_tsvector(
    'english',
    coalesce(item_name, '') || ' ' || coalesce(category, '') || ' ' || coalesce(description, '') || ' ' || coalesce(preferred_vendor, '')
)
WHERE search_vector IS NULL;

UPDATE vendors
SET search_vector = to_tsvector(
    'english',
    coalesce(vendor_name, '') || ' ' || coalesce(array_to_string(categories, ' '), '') || ' ' || coalesce(region, '')
)
WHERE search_vector IS NULL;

UPDATE policy_chunks
SET search_vector = to_tsvector(
    'english',
    coalesce(content, '') || ' ' || coalesce(policy_name, '') || ' ' || coalesce(object_name, '')
)
WHERE search_vector IS NULL;

-- Vector ANN indexes for faster semantic similarity search (ORDER BY embedding <=> query LIMIT K)
-- Using cosine distance because query path uses the <=> operator.
CREATE INDEX IF NOT EXISTS idx_inventory_items_embedding_cosine_ivfflat
ON inventory_items
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_inventory_items_fts
ON inventory_items
USING gin (search_vector);

CREATE INDEX IF NOT EXISTS idx_vendors_embedding_cosine_ivfflat
ON vendors
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_vendors_fts
ON vendors
USING gin (search_vector);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_embedding_cosine_ivfflat
ON policy_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_fts
ON policy_chunks
USING gin (search_vector);
