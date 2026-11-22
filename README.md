# clickhouse_vector_database — group12

Vector search benchmark on **ClickHouse**, following the *Common Baseline for the Vector Database Project*.

| Component | Version (pinned) |
| --- | --- |
| ClickHouse | `clickhouse/clickhouse-server:26.10.9.12` |
| Python | 3.10 (Ubuntu 22.04 package in the image) |
| Libraries | `clickhouse-connect==1.13.0`, `psutil==7.2.2` (`requirements.txt`) |
| Runtime | Docker (Compose v2) or Podman with the same flags |

## Run

```bash
docker compose build
docker compose up -d db
docker compose run --rm populate                  # DATASET=cohere-small-100k by default
docker compose run --rm benchmark db 1 443321 4
```

Pick the dataset with `DATASET`, e.g. `DATASET=glove-200-angular docker compose run --rm benchmark db 1 443321 4`.
Every step runs without prompts and returns a non-zero exit code on failure.

Off the course server, fetch the files first (prints size and sha256 of every file):

```bash
docker compose run --rm download                                         # $DATASET
docker compose run --rm download python3 /app/scripts/download.py all    # the five non-uform datasets
```

Everything for all five datasets (populate, goals 1/2/7, goal 7 file check, goal 8 at k = 10 and 100):

```bash
scripts/run_all.sh
```

## Datasets

Source: <https://huggingface.co/datasets/zuzaj/ADB>, files `train.csv`, `test.csv`, `ground_truth.csv` used unchanged.
Distance is cosine for all of them. Only train vectors are loaded; queries and ground truth are read from `/data` at
benchmark time.

| Dataset | Rows × dim | Test queries | True neighbors | Table |
| --- | --- | --- | --- | --- |
| `cohere-small-100k` | 100,000 × 768 | 1,000 | 1000 | `cohere_small_100k` |
| `nytimes-256-angular` | 290,000 × 256 | 10,000 | 1000 | `nytimes_256_angular` |
| `glove-200-angular` | 1,183,514 × 200 | 10,000 | 100 | `glove_200_angular` |
| `pubmed-arxiv` | 400,335 × 768 | 1,000 | 1000 | `pubmed_arxiv` |
| `ccnews-nomic-768-normalized` | 495,328 × 768 | 1,000 | 100 | `ccnews_nomic_768_normalized` |

`uform-wiki-1m` is skipped, as the baseline suggests: its ground truth has only 10 neighbors per query.

## Deployment

Two containers, as in section 2 of the baseline:

- **db** — ClickHouse server. Mounts `/data` read-only (all dataset files) and keeps the database in `/db`
  (named volume `db_data`). ClickHouse reads the dataset CSVs itself through `file('data/...')`.
- **client** — the `populate` and `benchmark` services (same image, Python scripts). They talk to ClickHouse over
  HTTP (port 8123), write CSVs to `/results`, and share the db container's PID namespace so the database's RSS can
  be sampled with `psutil`.

Resource limits come from environment variables. The defaults in `compose.yaml` are the proposed server limits:

| Variable | Server default | Local `.env.example` |
| --- | --- | --- |
| `DB_CPUSET` / `DB_MEM` | `0-3` / `12g` | `0-3` / `5g` |
| `CLIENT_CPUSET` / `CLIENT_MEM` | `4` / `2g` | `4` / `1g` |

Swap is off for both containers (`memswap_limit` equals `mem_limit`). For a small machine: `cp .env.example .env`.
On the server, do not create `.env`. Set `DATA_DIR` if the dataset files are not in `./data`.

## Database design

One MergeTree table per dataset, DDL in [`sql/ddl.sql`](sql/ddl.sql) in two marked parts:

1. **Schema** (before loading): `id UInt64` (0-based row position), `vector Array(Float32)` with a
   `length(vector) = d` constraint.
2. **Index** (after loading): `vector_similarity('hnsw', 'cosineDistance', d, 'f32', 32, 128)` — HNSW with f32
   vectors, M = 32, ef_construction = 128, granularity large enough for one index per part.

## Populate

`scripts/populate.py`, in the baseline order:

1. Schema: drop and recreate the table.
2. Load: `INSERT ... SELECT FROM file('train.csv')` with insert blocks of 50,000 rows, then `OPTIMIZE ... FINAL`
   to one part. Timed as `ingest_s`.
3. Index: add and materialize the HNSW index. Timed separately as `index_s`.
4. Checks: row count, dimension, and 10 random ids compared exactly with `train.csv`.

Peak RSS of the ClickHouse server is sampled every 100 ms during load and index build.

## Goals CLI

`db <goal> <params...>`, details and the query behind each goal in [`docs/goals.md`](docs/goals.md).

| Goal | Command | Status |
| --- | --- | --- |
| 1 | `db 1 <id> <k>` | k nearest neighbors of a stored vector |
| 2 | `db 2 <id> <radius>` | vectors within a cosine radius (at most 100) |
| 3 | `db 3 ...` | not implemented, not defined in the baseline yet |
| 4–6 | `db 4/5/6 ...` | not implemented, clustering is optional / undecided |
| 7 | `db 7 <id_a> <id_b>` | cosine distance between two stored vectors |
| 8 | `db 8 <n_queries> <k>` | benchmark: recall@k and latency over the first n test queries |

## Results

`results/build.csv` and `results/search.csv` use the baseline columns (section 7). Latency `p50/p95/p99_ms` is
measured by the client around the whole call; `db_p*_ms` is ClickHouse's own elapsed time for the query
(`elapsed_ns` from the `X-ClickHouse-Summary` response header). The numbers committed now come from a laptop
(AMD Ryzen 7 7840HS, WSL2, 4 db cores / 5 GB), not from the course server.

Goal 8 protocol: 200 warm-up queries, then the first `n` test queries 3 times per setting. Settings are
`hnsw_candidate_list_size_for_search` (ef_search) values, default `16,64,256`; `exact` runs a brute-force scan.
Override with `SETTINGS`, `REPEATS` and `WARMUP`, e.g. `SETTINGS=64,256,exact`.

## ClickHouse limitations

- The vector index works at granule level: it selects granules, then ClickHouse computes exact distances inside
  them. Recall and latency therefore also depend on `index_granularity`.
- One HNSW index per data part. Merges rebuild it, so the table is merged to one part before the index is built.
- `hnsw_candidate_list_size_for_search` below `k` appears to behave like `k`: on cohere, k = 100 gives the same
  recall at ef 16 and 64, so k = 100 sweeps need ef values above 100.
- Index build is CPU-heavy (M = 32): about 55 s for cohere and 170 s for nytimes on 4 cores.
- The whole index must be in memory to search. The vector index cache defaults to 5 GB, which matters with a
  12 GB container on the larger datasets.
