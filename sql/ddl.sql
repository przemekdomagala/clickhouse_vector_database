-- DDL for one dataset table. populate.py substitutes {table} and {dim}.
-- Table name = dataset file name with '-' replaced by '_', e.g. cohere_small_100k.

-- ===== PART 1: SCHEMA (run before loading) =====
DROP TABLE IF EXISTS {table} SYNC;

CREATE TABLE {table}
(
    id     UInt64,          -- 0-based row position in train.csv, matches ground_truth neighbor ids
    vector Array(Float32),  -- embedding of fixed length {dim}
    CONSTRAINT vector_dim CHECK length(vector) = {dim}
)
ENGINE = MergeTree
ORDER BY id;

-- ===== PART 2: INDEX (run after loading) =====
-- HNSW from ClickHouse's vector_similarity index:
--   distance        cosineDistance
--   dimensions      {dim}
--   quantization    f32 (no compression of the stored graph vectors)
--   M               32  (hnsw_max_connections_per_layer)
--   ef_construction 128 (hnsw_candidate_list_size_for_construction)
--   GRANULARITY     100000000, so one index covers a whole part (the table is merged to one part first)
-- Search-time parameter: hnsw_candidate_list_size_for_search (ef_search), set per query in goal 8.
ALTER TABLE {table}
    ADD INDEX vector_idx vector TYPE vector_similarity('hnsw', 'cosineDistance', {dim}, 'f32', 32, 128)
    GRANULARITY 100000000;

ALTER TABLE {table} MATERIALIZE INDEX vector_idx SETTINGS mutations_sync = 2;
