# Q&A — meeting preparation (group12, ClickHouse)

The numbers come from our local run (`results/build.csv`, `results/search.csv`), not from the course server.

---

## 1. Which indexes do we use, which others exist, and how are they stored?

**What we use:**
- **HNSW vector index** (`vector_similarity`), a MergeTree skip index:
  `vector_similarity('hnsw', 'cosineDistance', d, 'f32', 32, 128)`, i.e. cosine distance, f32 vectors in the
  graph, M = 32, ef_construction = 128. At search time we set `hnsw_candidate_list_size_for_search` (ef_search).
- **Primary key** `ORDER BY id`, a sparse index with one entry per granule (~8192 rows or 10 MB). It makes
  `WHERE id = ...` fast.

**Other indexes in ClickHouse** (checked in `system.data_skipping_index_types`):
- `minmax`, `set`, `bloom_filter`, `tokenbf_v1`, `ngrambf_v1`, `text` (full-text), `sparse_grams`,
  `hypothesis`; for vectors only `vector_similarity`.
- The only algorithm in `vector_similarity` is HNSW (usearch library). There is no IVF or DiskANN.
- Vector quantization in the index: `f64`, `f32`, `f16`, `bf16`, `i8`, `b1`.
- Without an index: an exact scan (brute force) with `ORDER BY cosineDistance(...) LIMIT k`. Newer options speed
  up that scan: the `QBit` type and the `Quantized` codec.

**How they are stored:**
- **On disk:** the HNSW index is a file in every table part (`skp_idx_vector_idx.idx`; for cohere 289 MB next to
  the 283 MB `vector.bin` column). There is one HNSW graph per part, which is why we merge the table into one part
  before building the index.
- **In RAM:** on the first query the whole index of a part is loaded into the *vector similarity index cache*.
  Its size is `vector_similarity_index_cache_size` (default 5 GiB), capped by `cache_size_to_ram_max_ratio`
  (default 0.5 × available RAM); for us that is 2.5 GiB with a 5 GB container limit. The eviction policy is SLRU
  (segmented LRU): new entries go to a probationary segment and move to a protected segment when used again, so
  one-off loads can't evict hot indexes. Graph search runs entirely in RAM. An index that doesn't fit in the cache
  would be re-read from disk on every query and would be very slow.
- **Partly in RAM:** the primary key (`primary.cidx`) is fully in RAM, marks are in the *mark cache*, and column
  data is read from disk or the page cache only for the granules a query needs.
- On the server with 12 GB the cap is 6 GB, so the 5 GiB default applies, while the largest index (ccnews) is
  1.35 GB. None of these are in our config; they are ClickHouse defaults.

## 2. Database design

- One MergeTree table per dataset, named with `-` replaced by `_` (e.g. `glove_200_angular`).
- Columns: `id UInt64` (row position in `train.csv`, matching `ground_truth`) and `vector Array(Float32)` with
  `CONSTRAINT vector_dim CHECK length(vector) = d`. `Array(Float32)` is the type the HNSW index requires;
  ClickHouse has no fixed-length vector type, so the constraint enforces the length.
- `ORDER BY id`, so fetching a vector by id reads a single granule.
- The DDL is in `sql/ddl.sql` in two marked parts:
  1. schema (before loading),
  2. index (`ADD INDEX` and `MATERIALIZE INDEX` after loading).
- No extra columns; goal 3 (`id mod 100`) can be computed in the query.

## 3. Populate results (hardware)

**Hardware:**
- Laptop with an AMD Ryzen 7 7840HS (8 cores / 16 threads), Windows + WSL2 with 7.7 GB RAM, laptop disk (WSL
  vhdx on C:).
- db container: 4 cores (`cpuset 0-3`) and 5 GB RAM, swap off.
- Client container: 1 core (`cpuset 4`) and 1 GB.
- Docker 26.1, ClickHouse 26.10.9.12.

| Dataset | Rows × dim | Load [s] | Index [s] | DB size [MB] | Index [MB] | Peak RAM [MB] |
| --- | --- | --- | --- | --- | --- | --- |
| cohere-small-100k | 100k × 768 | 5.0 | 55 | 546 | 275 | 1820 |
| nytimes-256-angular | 290k × 256 | 4.7 | 169 | 557 | 295 | 1266 |
| glove-200-angular | 1.18M × 200 | 31.3 | 723 | 1812 | 973 | 2495 |
| pubmed-arxiv | 400k × 768 | 31.5 | 244 | 2205 | 1117 | 3507 |
| ccnews-nomic-768-normalized | 495k × 768 | 43.1 | 315 | 2731 | 1384 | 3512 |

Notes:
- "Load" covers CSV parsing, inserting in batches of 50,000 rows, and `OPTIMIZE FINAL`. Downloading the files is
  not included.
- Building the index dominates the time and uses all 4 cores.
- Peak RAM is the sum of RSS of the ClickHouse processes, sampled every 100 ms. For pubmed and ccnews it also
  includes the indexes of earlier datasets that stayed in the cache, because we loaded the datasets one after
  another without restarting the server.
- Every dataset passed the checks: row count, dimension, and 10 random ids identical to the file.

## 4. Everything we did

- **Repository:** README (versions, commands, datasets, design, limitations), `docs/goals.md` describing each
  goal, `.gitignore` (data and database files are never committed).
- **Docker:** one `Dockerfile` (ClickHouse 26.10.9.12 + Python 3.10) and `compose.yaml` with services:
  - `db`: the server, `/data` read-only, `/db` as a volume;
  - `populate` and `benchmark`: the client;
  - `download`: only needed off the server.

  Limits are set with variables: the defaults are the server values (4 cores / 12 GB and 1 core / 2 GB), and
  locally `.env` lowers them.
- **Download:** fetches the files from Hugging Face and prints size and sha256.
- **Populate:** schema, load, index and checks; writes a row to `build.csv` with separate `ingest_s` and
  `index_s`.
- **`db` CLI:** goals 1, 2, 7 and 8; goals 3–6 are stubs that exit with code 2.
  - **Goal 1** (`db 1 <id> <k>`): finds the k vectors most similar to the stored vector with this id.
  - **Goal 2** (`db 2 <id> <radius>`): returns all vectors whose cosine distance from the stored vector is at
    most the radius, capped at the 100 nearest.
  - **Goal 3** (not implemented): k-NN restricted to a percentage of the rows (1, 10 or 50%); the baseline doesn't
    define it yet.
  - **Goal 4** (not implemented): groups all vectors into 100 clusters with spherical K-means and stores each
    vector's cluster id.
  - **Goal 5** (not implemented): returns the cluster id of a stored vector.
  - **Goal 6** (not implemented): finds the k most similar vectors within the same cluster as the given vector.
  - **Goal 7** (`db 7 <id_a> <id_b>`): prints the cosine distance between two stored vectors.
  - **Goal 8** (`db 8 <n_queries> <k>`): the benchmark; it runs the first n test queries, compares the results
    with the ground truth to get recall@k, and measures latency.
- **Validation:** `check_goal7.py` compares goal 7 with the file to 1e-5.
- **Run everything:** `run_all.sh`.
- **Results:** the full goal 8 protocol on 5 datasets (200 warm-up queries, 1000 queries × 3 repeats,
  ef 16/64/256, k = 10 and 100), i.e. 30 rows in `search.csv`.
- **Skipped:** uform-wiki-1m, as the baseline suggests.

## 5. How the client works

- The `benchmark` container runs the same image. The `db` program (`scripts/db.py`) in Python connects to
  ClickHouse over HTTP (port 8123) with the `clickhouse-connect` library.
- The dataset is chosen with the `DATASET` variable (default cohere); the table is the name with `_`.
- The query vector is passed as a parameter `{q:Array(Float32)}`. The HNSW index only works with a constant
  reference vector, so goal 1 first fetches the vector by id and only then searches for neighbors.
- Measurement:
  - the main latency is measured on the client side (`perf_counter`) around the whole HTTP call;
  - the database-side time is `elapsed_ns` from the `X-ClickHouse-Summary` response header, so no extra query;
  - the difference is about 3 ms: HTTP, sending the vector, and parsing.
- Queries run one after another, one at a time. The server may use up to 4 threads per query.
- The client shares the PID namespace with the db container (`pid: service:db`), so `psutil` can sample the
  server's RSS every 100 ms.
- Exit codes: 0 success, 1 runtime error (e.g. missing id), 2 bad arguments or unimplemented goal.

## 6. Populating

The order from the baseline:
1. **Schema:** `DROP TABLE` and `CREATE`, so a re-run starts from scratch.
2. **Load:** `INSERT INTO t SELECT id, vector FROM file('data/<dataset>/train.csv', 'CSVWithNames', ...)`
   with `max_insert_block_size = min_insert_block_size_rows = 50000`. ClickHouse reads the CSV from `/data`
   itself, as a stream in batches of 50k rows, so the file never has to fit in RAM.
   Then `OPTIMIZE TABLE FINAL` merges the table into one part. All of this is `ingest_s`.
3. **Index:** `ADD INDEX` and `MATERIALIZE INDEX` with `mutations_sync = 2` (we wait until the build finishes).
   Measured separately as `index_s`.
4. **Checks:**
   - row count equals the number in the manifest and the baseline;
   - min and max `length(vector)` equal d;
   - 10 random ids (seed 42) have vectors bit-for-bit identical to `train.csv` (a JOIN of the table with
     `file()`).

   Any failed check gives a non-zero exit code.
5. RSS is sampled throughout loading and index build, then a row is written to `build.csv`.

Test queries and ground truth are not loaded into the database; goal 8 reads them from `/data` at benchmark time.

## 7. Tools

- **Docker 26.1 + Compose v2** (compatible with Podman). Limits: `cpuset`, `mem_limit`, and `memswap_limit`
  equal to `mem_limit`, i.e. no swap.
- **ClickHouse 26.10.9.12** (`clickhouse/clickhouse-server` image, pinned). 26.11 has no published image.
- **Python 3.10**, `clickhouse-connect==1.13.0` (HTTP client), `psutil==7.2.2` (RSS); versions in
  `requirements.txt`.
- **Scripts:**
  - `download.py`, `populate.py`;
  - `db.py` (CLI), `check_goal7.py`;
  - `run_all.sh` (everything for the 5 datasets);
  - `common.py` (dataset metadata, RAM monitor, CSV writing).
- **Results:** `results/build.csv` and `results/search.csv` with the baseline columns (section 7).
- Non-interactive: everything runs with the standard `docker compose ...` commands, without prompts, and with a
  non-zero exit code on failure.

## 8. Which queries can be run in the current client

| Goal | Command | Parameters | Output |
| --- | --- | --- | --- |
| 1 | `db 1 <id> <k>` | id 0..rows-1, k 1..100 | k lines `id distance`, nearest first, without the vector itself |
| 2 | `db 2 <id> <radius>` | radius 0..2 | lines `id distance` with distance ≤ radius, at most 100; result count on stderr |
| 7 | `db 7 <id_a> <id_b>` | two ids | one number, the cosine distance |
| 8 | `db 8 <n_queries> <k>` | n 1..number of test queries, k 1..100 | one line per setting: recall@k, p50/p95/p99, db_p50/p95/p99, peak RAM; rows in `search.csv` |
| 3–6 | `db 3..6 ...` | — | "not implemented", code 2 |

Examples:

```bash
docker compose run --rm benchmark db 1 1000 10
DATASET=glove-200-angular docker compose run --rm benchmark db 2 1000 0.563
docker compose run --rm benchmark db 7 1000 98999
```

Goal 8 has options as environment variables: `SETTINGS` (list of ef_search values or `exact`, default
`16,64,256`), `REPEATS` (3) and `WARMUP` (200), e.g.
`REPEATS=1 WARMUP=20 SETTINGS=16,256,exact docker compose run --rm benchmark db 8 100 10`.

Arbitrary SQL: `docker compose exec db clickhouse-client`.

## 9. Implemented goals

- **1 (kNN):** fetch the vector by id, then `ORDER BY cosineDistance(vector, q) LIMIT k+1` through the HNSW
  index; drop the row with the query id itself and return k results.
- **2 (radius):** the same with `LIMIT 101`, then a `distance ≤ radius` filter. We don't use
  `WHERE cosineDistance <= r`, because the index only serves `ORDER BY ... LIMIT`; a WHERE filter would force a
  full scan. The 100-result cap comes from the baseline.
- **7 (distance):** `cosineDistance` of two vectors fetched by id.
- **8 (benchmark):**
  - test queries and ground truth (`rank < k`) are read from the files outside the timed part;
  - per setting: 200 warm-up queries, then n queries × 3;
  - recall@k = |returned ∩ true neighbors| / k, averaged.
- **3:** not implemented, because the baseline doesn't define it (there is only a `<percent>` parameter).
- **4–6:** not implemented, because clustering is optional and undecided in the baseline.

### Re 8, 9: what to expect and how to validate the results

**Expected goal 8 results** (client p50 latency / recall@10):

| Dataset | ef 16 | ef 64 | ef 256 |
| --- | --- | --- | --- |
| cohere | 10.5 ms / 0.885 | 11.9 ms / **0.979** | 14.4 ms / 0.996 |
| nytimes | 8.8 ms / 0.775 | 9.6 ms / 0.876 | 12.2 ms / 0.928 |
| glove | 7.6 ms / 0.567 | 8.4 ms / 0.753 | 11.4 ms / 0.873 |
| pubmed | 10.6 ms / 0.895 | 11.0 ms / **0.983** | 13.4 ms / 0.996 |
| ccnews | 9.1 ms / 0.921 | 10.7 ms / **0.982** | 13.3 ms / 0.997 |

`exact` gives recall 1.0 at about 420 ms per query on cohere (full scan on 4 cores).

**Validation:**
- **Goal 7:** `docker compose run --rm benchmark python3 /app/scripts/check_goal7.py` computes the distance
  independently, in Python with float64, from the vectors as written in `train.csv`, for 100 pairs
  (id, 99999 − id). The maximum difference was about 1e-7 on every dataset (1e-5 required).
- **Goal 1:** compare with an exact scan without the index:
  ```bash
  docker compose exec db clickhouse-client -q "WITH (SELECT vector FROM cohere_small_100k WHERE id = 1000) AS q
    SELECT id, cosineDistance(vector, q) AS d FROM cohere_small_100k WHERE id != 1000
    ORDER BY d LIMIT 10 SETTINGS use_skip_indexes = 0"
  ```
  The result should match `db 1 1000 10`, apart from any differences caused by the HNSW approximation (on
  cohere for id 1000 the ids are identical).
- **Goal 2:** all distances ≤ radius, sorted ascending, at most 100 rows; exact scan as above with
  `WHERE d <= radius`.
- **Goal 8:**
  - recall is computed directly from `ground_truth.csv`;
  - sanity check: `SETTINGS=exact` must give recall 1.0, and on cohere it gave 1.0000;
  - recall increases with ef_search, and so does latency.
- **Populate:** built-in checks (row count, dimension, 10 vectors identical to the file).

---

## 10. Other questions the instructor may ask

**Why don't nytimes and glove reach recall@10 ≥ 0.95?**
- With M = 32 / ef_construction = 128 and ef_search ≤ 256 the HNSW graph isn't enough. Glove (angular) is a
  known hard dataset: the vectors are clustered and neighbors have very similar distances.
- The fix: larger ef_search (512–2000), larger M (48–64) or larger ef_construction, at the cost of build time and
  memory. That tuning is planned for M5.

**Why do ef 16 and 64 give identical recall at k = 100?**
- It looks like ClickHouse raises ef_search to at least k, because HNSW has to return k candidates. We observed
  this on all 5 datasets, so for k = 100 only values above 100 make sense.

**Why do you merge the table into one part before building the index?**
- HNSW is built separately for each MergeTree part. Many parts mean many small graphs: each one has to be
  searched, and every merge of parts rebuilds the graph from scratch.
- One part gives one graph, built once. Granularity 100000000 makes one index cover the whole part.

**What happens when new data is inserted after the index is built?**
- The new part gets its own index at insert time, and a query searches the indexes of all parts.
- Background merges rebuild the graph, which costs a lot of CPU. That's a ClickHouse limitation with frequent
  writes.

**Why pass the vector as a parameter and not as a subquery?**
- The `vector_similarity` index is only used when the reference vector is a constant.
- We checked with `EXPLAIN indexes = 1`: with a parameter the `vector_idx` index is used (it selects 9 of 29
  granules on cohere).

**Are the HNSW results re-scored exactly (rescoring)?**
- No: `vector_search_with_rescoring = 0` (the default), so the result and the distance come from the HNSW graph.
- ClickHouse reads from disk only the granules containing the rows it found. That's why recall depends on
  ef_search.

**Why two latencies, and which one matters more?**
- The main one is the client latency (end to end, including HTTP), as in the revised baseline, because times
  inside different databases aren't necessarily measured the same way.
- `db_p*` is the time reported by ClickHouse; the ~3 ms difference is the overhead of HTTP and sending the vector.

**How do you know the RAM measurement is correct?**
- Every 100 ms `psutil` sums the RSS of all `clickhouse*` processes in the db container (shared PID namespace).
- Limitation: RSS also includes caches (e.g. other datasets' indexes), so it's an upper bound.

**Why ClickHouse and not a dedicated vector database?**
- It's an analytical column store with an HNSW index: vectors can be combined with filtering, aggregation and
  SQL.
- Weaker points: one algorithm (HNSW), the index must fit in RAM, rebuilds on merges, and granularity at the level
  of granules and parts.

**How much memory does the index need, and what does it depend on?**
- Roughly n × d × 4 B (f32 vectors in the graph) plus the graph edges (n × M × ~2 × 4 B).
- Cohere: 275 MB of index for 293 MB of raw vectors; ccnews: 1.35 GB.
- `bf16`/`i8` quantization shrinks it 2–4× at the cost of recall.

**What is HNSW?**
- A multi-layer "small world" graph: the upper layers are sparse (long jumps), the bottom layer contains all
  points.
- Greedy search descends layer by layer. M is the number of edges per node; ef_construction and ef_search are the
  candidate list sizes during build and search.

**Why does goal 2 return few results (cohere 44, ccnews 0 for id 1000)?**
- The radius is the median distance to the 10th neighbor for the test queries, but we query with a train vector,
  whose neighborhood may be sparser. That's a property of the definition, not a bug.
- We return at most the 100 nearest from the index, as the baseline says.

**Can you do filtered search (goal 3)?**
- Yes: `WHERE id % 100 < percent` with `ORDER BY cosineDistance LIMIT k`.
- `vector_search_filter_strategy` chooses pre-filtering (WHERE first, then an exact scan) or post-filtering (index
  first, then the filter, with `vector_search_index_fetch_multiplier`).
- At 1% of the data, pre-filtering is better.

**Why does the server read the data (`file()`) and not Python?**
- It's faster: ClickHouse's CSV parser is parallel, while in Python we'd have to parse 5 GB of text.
- It streams in batches of 50k, without holding the file in client memory.
- `/data` is mounted only in the db container, as in the baseline.

**How do you ensure reproducibility?**
- Pinned versions (image, libraries) and a fixed seed (42) for the random ids in the checks.
- Deterministic queries (the first n test queries) and the commands in the README.
- One command, `scripts/run_all.sh`, reproduces everything.

**What counts towards load time?**
- From the start of `INSERT` to the end of `OPTIMIZE FINAL`: CSV parsing, writing and merging. The index build is
  `index_s`, separately.
- The baseline is inconsistent here: §3 says measure them separately, §6 says together. We report them separately,
  and the sum is easy to compute.

**What if someone passes bad parameters?**
- Range validation (id, k ≤ min(100, true neighbors), radius 0..2, n ≤ number of queries) and exit code 2.
- An id missing from the database gives code 1.

**How do you move this to the course server?**
- No `.env`, so the default limits apply: 12 GB / 4 cores and 2 GB / 1 core.
- `DATA_DIR` points to the directory with the files, then the four standard commands from the README.
- `download` isn't needed, because the files are already there.
