#!/usr/bin/env python3
"""db <goal> <params...> -- the goals CLI from section 8 of the baseline.

The dataset is selected with the DATASET environment variable (default cohere-small-100k).
Goal 8 options (environment): SETTINGS (comma list of ef_search values and/or 'exact', default 16,64,256),
REPEATS (default 3), WARMUP (default 200).
"""
import os
import sys
import time

from common import (DATASETS, GROUP, SEARCH_COLUMNS, SYSTEM, Monitor, UsageError, append_csv, client,
                    dataset_name, file_source, percentile, table_name)

USAGE = """usage: db <goal> <params...>
  db 1 <id> <k>              k nearest neighbors of a stored vector
  db 2 <id> <radius>         vectors within a cosine distance radius (at most 100)
  db 7 <id_a> <id_b>         cosine distance between two stored vectors
  db 8 <n_queries> <k>       benchmark: first n test queries, recall@k and latency
  goals 3, 4, 5, 6 are not implemented (optional / undecided in the baseline)
dataset: DATASET=<name> (default cohere-small-100k)"""

KNN_SQL = """SELECT id, cosineDistance(vector, {{q:Array(Float32)}}) AS distance FROM {table}
ORDER BY cosineDistance(vector, {{q:Array(Float32)}}) LIMIT {limit}"""
IDS_SQL = "SELECT id FROM {table} ORDER BY cosineDistance(vector, {{q:Array(Float32)}}) LIMIT {limit}"


def parse_int(value, name, lo, hi):
    try:
        n = int(value)
    except ValueError:
        raise UsageError(f"<{name}> must be an integer, got {value!r}")
    if not lo <= n <= hi:
        raise UsageError(f"<{name}> must be in {lo}..{hi}, got {n}")
    return n


def parse_k(value, meta):
    return parse_int(value, "k", 1, min(100, meta["neighbors"]))


def parse_id(value, meta, name="id"):
    return parse_int(value, name, 0, meta["rows"] - 1)


def stored_vector(c, table, vid):
    rows = c.query(f"SELECT vector FROM {table} WHERE id = {vid}").result_rows
    if not rows:
        raise RuntimeError(f"id {vid} not found in {table}")
    return rows[0][0]


def neighbors_of(c, table, vid, limit):
    q = stored_vector(c, table, vid)
    rows = c.query(KNN_SQL.format(table=table, limit=limit + 1), parameters={"q": q}).result_rows
    return [(i, d) for i, d in rows if i != vid][:limit]


def print_pairs(rows):
    for i, d in rows:
        print(f"{i} {d:.9g}")


def goal_1(c, table, meta, args):
    if len(args) != 2:
        raise UsageError("db 1 <id> <k>")
    vid, k = parse_id(args[0], meta), parse_k(args[1], meta)
    print_pairs(neighbors_of(c, table, vid, k))


def goal_2(c, table, meta, args):
    if len(args) != 2:
        raise UsageError("db 2 <id> <radius>")
    vid = parse_id(args[0], meta)
    try:
        radius = float(args[1])
    except ValueError:
        raise UsageError(f"<radius> must be a number, got {args[1]!r}")
    if not 0 <= radius <= 2:
        raise UsageError(f"<radius> must be in 0..2, got {radius}")
    rows = [(i, d) for i, d in neighbors_of(c, table, vid, 100) if d <= radius]
    print_pairs(rows)
    print(f"results: {len(rows)}", file=sys.stderr)


def goal_7(c, table, meta, args):
    if len(args) != 2:
        raise UsageError("db 7 <id_a> <id_b>")
    a, b = parse_id(args[0], meta, "id_a"), parse_id(args[1], meta, "id_b")
    found = c.command(f"SELECT count() FROM {table} WHERE id IN ({a}, {b})")
    if found != len({a, b}):
        raise RuntimeError(f"ids {a}, {b} not both found in {table}")
    d = c.command(f"""SELECT cosineDistance((SELECT vector FROM {table} WHERE id = {a}),
                                           (SELECT vector FROM {table} WHERE id = {b}))""")
    print(f"{float(d):.9g}")


def search_settings():
    raw = os.environ.get("SETTINGS", "").strip()
    out = []
    for token in (raw.split(",") if raw else ["16", "64", "256"]):
        token = token.strip()
        if token == "exact":
            out.append(("exact", "", {"use_skip_indexes": 0}))
        else:
            ef = parse_int(token, "SETTINGS entry", 1, 100_000)
            out.append(("hnsw_candidate_list_size_for_search", ef, {"hnsw_candidate_list_size_for_search": ef}))
    return out


def goal_8(c, table, meta, args):
    if len(args) != 2:
        raise UsageError("db 8 <n_queries> <k>")
    dataset = os.environ.get("DATASET", "cohere-small-100k")
    n = parse_int(args[0], "n_queries", 1, meta["queries"])
    k = parse_k(args[1], meta)
    repeats = parse_int(os.environ.get("REPEATS") or "3", "REPEATS", 1, 100)
    warmup = parse_int(os.environ.get("WARMUP") or "200", "WARMUP", 0, meta["queries"])
    settings = search_settings()

    queries = c.query(f"""SELECT vector FROM {file_source(dataset, 'test.csv', 'id UInt64, vector Array(Float32)')}
                          WHERE id < {max(n, warmup)} ORDER BY id""").result_rows
    queries = [r[0] for r in queries]
    truth = c.query(f"""SELECT query_id, groupArray(neighbor_id) FROM (
            SELECT query_id, neighbor_id FROM {file_source(dataset, 'ground_truth.csv',
                'query_id UInt64, rank UInt32, neighbor_id UInt64, distance Float32')}
            WHERE query_id < {n} AND rank < {k} ORDER BY query_id, rank)
        GROUP BY query_id ORDER BY query_id""").result_rows
    truth = [set(ids) for _, ids in truth]
    if len(queries) < n or len(truth) != n:
        raise RuntimeError(f"expected {n} test queries and ground-truth rows, got {len(queries)} / {len(truth)}")

    version = c.command("SELECT version()")
    sql = IDS_SQL.format(table=table, limit=k)
    for name, value, ch_settings in settings:
        with Monitor() as mon:
            for q in queries[:warmup]:
                c.query(sql, parameters={"q": q}, settings=ch_settings)
            e2e, db_side, hits = [], [], 0
            for _ in range(repeats):
                hits = 0
                for i in range(n):
                    t = time.perf_counter()
                    r = c.query(sql, parameters={"q": queries[i]}, settings=ch_settings)
                    e2e.append((time.perf_counter() - t) * 1000)
                    db_side.append(int(r.summary.get("elapsed_ns", 0)) / 1e6)
                    hits += len(truth[i].intersection(row[0] for row in r.result_rows))
        recall = hits / (n * k)
        row = dict(group=GROUP, system=SYSTEM, version=version, dataset=dataset, k=k, setting_name=name,
                   setting_value=value, recall=round(recall, 4),
                   **{f"p{p}_ms": round(percentile(e2e, p), 3) for p in (50, 95, 99)},
                   **{f"db_p{p}_ms": round(percentile(db_side, p), 3) for p in (50, 95, 99)})
        append_csv("search.csv", SEARCH_COLUMNS, row)
        label = f"{name}={value}" if value != "" else name
        print(f"{label} recall@{k}={recall:.4f} p50={row['p50_ms']}ms p95={row['p95_ms']}ms "
              f"p99={row['p99_ms']}ms db_p50={row['db_p50_ms']}ms db_p95={row['db_p95_ms']}ms "
              f"db_p99={row['db_p99_ms']}ms queries={n * repeats} peak_mem_mb={mon.peak_mb}")


def not_implemented(goal):
    def run(*_):
        raise UsageError(f"goal {goal} is not implemented (optional / undecided in the baseline)")
    return run


GOALS = {"1": goal_1, "2": goal_2, "3": not_implemented(3), "4": not_implemented(4), "5": not_implemented(5),
         "6": not_implemented(6), "7": goal_7, "8": goal_8}


def main(argv):
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in GOALS:
        print(USAGE, file=sys.stderr)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    try:
        dataset = dataset_name()
        GOALS[argv[0]](client(), table_name(dataset), DATASETS[dataset], argv[1:])
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
