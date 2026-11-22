#!/usr/bin/env python3
"""Fetch dataset files from Hugging Face into /data/<dataset>/ and print size + sha256.

Usage: download.py [dataset ...|all]   (default: $DATASET)
Only needed off the course server; there the files are already present.
"""
import hashlib
import json
import os
import sys
import urllib.request

from common import DATASETS, DEFAULT_DATASETS

BASE = "https://huggingface.co/datasets/zuzaj/ADB/resolve/main"
TREE = "https://huggingface.co/api/datasets/zuzaj/ADB/tree/main"
DATA = os.environ.get("DATA_DIR", "/data")
FILES = ["manifest.json", "train.csv", "test.csv", "ground_truth.csv"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, path):
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
    os.replace(tmp, path)


def remote_sizes(dataset):
    # manifest.json output_bytes is not always accurate (ccnews ground_truth.csv), so trust the hub listing
    with urllib.request.urlopen(f"{TREE}/{dataset}", timeout=60) as r:
        return {e["path"].rsplit("/", 1)[-1]: e["size"] for e in json.load(r) if e["type"] == "file"}


def download(dataset):
    target = os.path.join(DATA, dataset)
    os.makedirs(target, exist_ok=True)
    expected = remote_sizes(dataset)
    for name in FILES:
        path = os.path.join(target, name)
        if not (os.path.exists(path) and os.path.getsize(path) == expected.get(name)):
            print(f"{dataset}/{name}: downloading", flush=True)
            fetch(f"{BASE}/{dataset}/{name}", path)
        size = os.path.getsize(path)
        if size != expected.get(name):
            raise SystemExit(f"{dataset}/{name}: size {size} != hub size {expected.get(name)}")
        print(f"{dataset}/{name}  {size} bytes  sha256 {sha256(path)}", flush=True)


def main():
    os.umask(0)
    names = sys.argv[1:] or [os.environ.get("DATASET", "cohere-small-100k")]
    if names == ["all"]:
        names = DEFAULT_DATASETS
    for name in names:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset {name!r}")
        download(name)


if __name__ == "__main__":
    main()
