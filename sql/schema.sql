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
