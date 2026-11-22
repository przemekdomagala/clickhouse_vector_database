#!/usr/bin/env python3
"""Check goal 7 against train.csv: pairs (id, 99999 - id) for the test ids 0, 1000, ..., 99000.

The reference distance is computed here in float64 from the vectors as written in the file,
independently of ClickHouse; every pair must match `db 7` to 1e-5.
"""
import contextlib
import io
import json
import math
import sys

from common import client, dataset_name, file_source, table_name, DATASETS
from db import goal_7

TOLERANCE = 1e-5


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return 1 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    dataset = dataset_name()
    table, meta = table_name(dataset), DATASETS[dataset]
    c = client()
    pairs = [(i, 99_999 - i) for i in range(0, 100_000, 1000)]
    ids = sorted({x for p in pairs for x in p})
    rows = c.query(f"""SELECT id, vector FROM {file_source(dataset, 'train.csv', 'id UInt64, vector String')}
                       WHERE id IN {tuple(ids)}""").result_rows
    vectors = {i: json.loads(v) for i, v in rows}

    worst = 0.0
    for a, b in pairs:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            goal_7(c, table, meta, [str(a), str(b)])
        diff = abs(float(out.getvalue()) - cosine(vectors[a], vectors[b]))
        worst = max(worst, diff)
    print(f"{dataset}: {len(pairs)} pairs, max |db 7 - file| = {worst:.3g} (tolerance {TOLERANCE})")
    return 0 if worst <= TOLERANCE else 1


if __name__ == "__main__":
    sys.exit(main())
