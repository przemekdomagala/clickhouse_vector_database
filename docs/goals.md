# Goals (section 8 of the baseline)

All goals: `db <goal> <params...>`, dataset from `DATASET` (default `cohere-small-100k`), table `<dataset with _>`.
Neighbor goals leave out the query vector itself and print `id distance`, nearest first.
Exit codes: 0 success, 1 runtime failure (e.g. id not in the table), 2 bad arguments or unimplemented goal.

Parameter checks: `k` in 1..min(100, true neighbors), `id` in 0..rows-1, `radius` in 0..2,
`n_queries` in 1..number of test queries.

## Goal 1 — `db 1 <id> <k>`

k nearest neighbors of a stored vector. The vector is read first, then sent as a query parameter so the HNSW
index can be used (it needs a constant reference vector):

```sql
SELECT vector FROM t WHERE id = <id>;
SELECT id, cosineDistance(vector, {q:Array(Float32)}) AS distance FROM t
ORDER BY cosineDistance(vector, {q:Array(Float32)}) LIMIT <k + 1>;
```

The row with the query id is dropped and the first `k` remaining rows are printed. Measured with k = 10 and 100.

## Goal 2 — `db 2 <id> <radius>`

All vectors within `radius` of a stored vector, at most 100. Same query as goal 1 with `LIMIT 101`, then rows with
`distance <= radius` are kept. A `WHERE cosineDistance(...) <= r` predicate would force a full scan, because the
index only serves `ORDER BY distance LIMIT n`. The number of results is printed to stderr.

Radii (median distance of the 10th true neighbor): cohere 0.142, nytimes 0.604, glove 0.563, pubmed 0.269,
ccnews 0.198.

## Goal 3

Not implemented. The baseline lists `<percent>` (1, 10, 50) but does not define the goal.

## Goals 4, 5, 6

Not implemented. Clustering is optional and still to be decided in the baseline. If kept: spherical K-means
`db 4 100 20` on cohere (ids 0..99 as initial centers, unit-normalized vectors and centers, exactly 20 iterations),
`db 5 <id>` cluster of a vector, `db 6 <id> <k>` k-NN inside that cluster.

## Goal 7 — `db 7 <id_a> <id_b>`

```sql
SELECT cosineDistance((SELECT vector FROM t WHERE id = <a>), (SELECT vector FROM t WHERE id = <b>));
```

Prints one number. `scripts/check_goal7.py` compares the pairs `(id, 99999 - id)` for ids 0, 1000, ..., 99000
with a float64 distance computed from `train.csv`; all must match to 1e-5.

## Goal 8 — `db 8 <n_queries> <k>`

The benchmark query. The first `n` test vectors and their ground truth (`rank < k`) are read from `/data`
through ClickHouse `file()`, outside the timed part. Per search setting:

1. 200 warm-up queries (not measured).
2. The first `n` queries, 3 times; each query is timed by the client around the whole HTTP call, and ClickHouse's
   `elapsed_ns` is recorded as the database-side time.
3. recall@k = overlap of returned ids with the true neighbors / k, averaged over the queries.

```sql
SELECT id FROM t ORDER BY cosineDistance(vector, {q:Array(Float32)}) LIMIT <k>
SETTINGS hnsw_candidate_list_size_for_search = <ef>;   -- or use_skip_indexes = 0 for exact
```

One row per setting is appended to `results/search.csv`. Measured with n = 1000 and k = 10 and 100.
