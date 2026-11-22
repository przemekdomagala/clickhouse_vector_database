import csv
import datetime
import os
import threading
import time

import clickhouse_connect
import psutil

GROUP = os.environ.get("GROUP", "group12")
SYSTEM = "clickhouse"
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")

# Section 1 (rows, dim, test queries, true neighbors per query) and section 8 (goal 2 radius).
DATASETS = {
    "cohere-small-100k":           dict(rows=100_000,   dim=768, queries=1_000,   neighbors=1000, radius=0.142),
    "nytimes-256-angular":         dict(rows=290_000,   dim=256, queries=10_000,  neighbors=1000, radius=0.604),
    "glove-200-angular":           dict(rows=1_183_514, dim=200, queries=10_000,  neighbors=100,  radius=0.563),
    "pubmed-arxiv":                dict(rows=400_335,   dim=768, queries=1_000,   neighbors=1000, radius=0.269),
    "ccnews-nomic-768-normalized": dict(rows=495_328,   dim=768, queries=1_000,   neighbors=100,  radius=0.198),
    "uform-wiki-1m":               dict(rows=1_000_000, dim=256, queries=100_000, neighbors=10,   radius=None),
}
DEFAULT_DATASETS = [d for d in DATASETS if d != "uform-wiki-1m"]

BUILD_COLUMNS = ["group", "system", "version", "dataset", "ingest_s", "index_s", "index_type", "db_mb",
                 "index_mb", "peak_mem_mb", "cpu", "cores", "ram_gb", "date", "kmeans_s"]
SEARCH_COLUMNS = ["group", "system", "version", "dataset", "k", "setting_name", "setting_value", "recall",
                  "p50_ms", "p95_ms", "p99_ms", "db_p50_ms", "db_p95_ms", "db_p99_ms"]


class UsageError(Exception):
    pass


def dataset_name():
    name = os.environ.get("DATASET", "cohere-small-100k")
    if name not in DATASETS:
        raise UsageError(f"unknown DATASET {name!r}; expected one of: {', '.join(DATASETS)}")
    return name


def table_name(dataset):
    return dataset.replace("-", "_")


def file_source(dataset, file, structure):
    return f"file('data/{dataset}/{file}', 'CSVWithNames', '{structure}')"


def client():
    return clickhouse_connect.get_client(
        host=os.environ.get("CH_HOST", "localhost"),
        port=int(os.environ.get("CH_PORT", "8123")),
        send_receive_timeout=36_000,
        compress=False,
    )


def percentile(values, p):
    s = sorted(values)
    pos = (len(s) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def append_csv(name, columns, row):
    path = os.path.join(RESULTS_DIR, name)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if new:
            w.writeheader()
        w.writerow(row)
    os.chmod(path, 0o666)


def today():
    return datetime.date.today().isoformat()


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def server_info(c):
    version = c.command("SELECT version()")
    cores = c.command("SELECT value FROM system.settings WHERE name = 'max_threads'")
    cores = "".join(ch for ch in str(cores) if ch.isdigit())
    ram = c.query("""SELECT metric, value FROM system.asynchronous_metrics
                     WHERE metric IN ('CGroupMemoryTotal', 'OSMemoryTotal')""").result_rows
    ram = dict(ram)
    ram_bytes = ram.get("CGroupMemoryTotal") or ram.get("OSMemoryTotal") or 0
    return dict(version=version, cores=cores, ram_gb=round(ram_bytes / 2**30, 1))


class Monitor:
    """Peak RSS summed over the ClickHouse server processes, sampled every 100 ms.

    Requires the client container to share the db container's PID namespace (pid: service:db).
    """

    def __init__(self, interval=0.1):
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def _sample(self):
        total = 0
        for p in psutil.process_iter(["name"]):
            try:
                if (p.info["name"] or "").startswith("clickhouse"):
                    total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._sample())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self):
        return round(self.peak / 2**20, 1)
