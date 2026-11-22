#!/usr/bin/env python3
"""Populate one dataset (section 6): schema -> batch load -> index -> checks; appends a row to build.csv.

Usage: populate.py [dataset]   (default: $DATASET)
"""
import json
import random
import re
import sys
import time

from common import (BUILD_COLUMNS, DATASETS, GROUP, SYSTEM, Monitor, append_csv, client, cpu_model,
                    dataset_name, file_source, server_info, table_name, today)

BATCH_ROWS = 50_000
TRAIN_STRUCTURE = "id UInt64, vector Array(Float32)"


def ddl_parts(table, dim):
    text = open("/app/sql/ddl.sql").read().replace("{table}", table).replace("{dim}", str(dim))
    schema, index = re.split(r"^-- ===== PART 2: INDEX.*$", text, flags=re.M)

    def statements(part):
        body = "\n".join(line for line in part.splitlines() if not line.strip().startswith("--"))
        return [s.strip() for s in body.split(";") if s.strip()]

    return statements(schema), statements(index)


def check(cond, message):
    if not cond:
        raise SystemExit(f"CHECK FAILED: {message}")
    print(f"check ok: {message}")


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else dataset_name()
    meta = DATASETS[dataset]
    table = table_name(dataset)
    c = client()

    manifest = json.loads(c.query(f"SELECT * FROM file('data/{dataset}/manifest.json', 'RawBLOB')").result_rows[0][0])
    check(manifest["train_rows"] == meta["rows"] and manifest["dimension"] == meta["dim"],
          f"manifest matches baseline ({meta['rows']} x {meta['dim']})")

    schema, index = ddl_parts(table, meta["dim"])
    for stmt in schema:
        c.command(stmt)
    print(f"schema created: {table}")

    with Monitor() as mon:
        t0 = time.perf_counter()
        c.command(
            f"INSERT INTO {table} SELECT id, vector FROM {file_source(dataset, 'train.csv', TRAIN_STRUCTURE)}",
            settings=dict(max_insert_block_size=BATCH_ROWS, min_insert_block_size_rows=BATCH_ROWS,
                          min_insert_block_size_bytes=0))
        # one part per table, so the HNSW index is built once over all vectors
        c.command(f"OPTIMIZE TABLE {table} FINAL")
        ingest_s = time.perf_counter() - t0
        print(f"loaded in {ingest_s:.1f} s (batches of {BATCH_ROWS} rows)")

        t1 = time.perf_counter()
        for stmt in index:
            c.command(stmt)
        index_s = time.perf_counter() - t1
        print(f"index built in {index_s:.1f} s")

    rows = c.command(f"SELECT count() FROM {table}")
    check(rows == meta["rows"], f"row count {rows} == {meta['rows']}")
    lo, hi = c.query(f"SELECT min(length(vector)), max(length(vector)) FROM {table}").result_rows[0]
    check(lo == hi == meta["dim"], f"dimension {lo}..{hi} == {meta['dim']}")
    ids = sorted(random.Random(42).sample(range(meta["rows"]), 10))
    matched = c.command(f"""
        SELECT count() FROM {table} AS t
        INNER JOIN (SELECT id, vector FROM {file_source(dataset, 'train.csv', TRAIN_STRUCTURE)}
                    WHERE id IN {tuple(ids)}) AS f ON t.id = f.id
        WHERE t.vector = f.vector""")
    check(matched == len(ids), f"{matched}/{len(ids)} random ids match train.csv vectors exactly")

    db_bytes = c.command(f"SELECT sum(bytes_on_disk) FROM system.parts WHERE active AND table = '{table}'")
    index_bytes = c.command(
        f"SELECT sum(data_compressed_bytes) FROM system.data_skipping_indices WHERE table = '{table}'")
    info = server_info(c)
    row = dict(group=GROUP, system=SYSTEM, version=info["version"], dataset=dataset,
               ingest_s=round(ingest_s, 2), index_s=round(index_s, 2), index_type="hnsw(f32,M=32,ef_c=128)",
               db_mb=round(db_bytes / 2**20, 1), index_mb=round(index_bytes / 2**20, 1),
               peak_mem_mb=mon.peak_mb, cpu=cpu_model(), cores=info["cores"], ram_gb=info["ram_gb"],
               date=today(), kmeans_s="")
    append_csv("build.csv", BUILD_COLUMNS, row)
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
