#!/usr/bin/env python3
"""Prepare the MS MARCO V1 (OpenAI ada-002) dataset for the Ada-ef experiments.

Standalone, path-robust version of the "MS MARCO V1" cell in
``data_prep.ipynb``. It:

  1. (optionally) downloads ``msmarco-passage-openai-ada2.tar`` into the
     dataset directory and extracts it,
  2. loads the 89 passage-embedding shards (``0.jsonl.gz`` .. ``88.jsonl.gz``)
     and the query embeddings
     (``topics.msmarco-passage.dev-subset.openai-ada2.jsonl.gz``),
  3. computes k=1000 ground-truth neighbors with hnswlib's brute-force cosine
     index,
  4. writes ``msmarco.hdf5`` (datasets: ``train``, ``test``, ``neighbors``)
     into ``$ADA_EF_ROOT/experiments/data`` (or ``../experiments/data``).

Unlike the notebook cell, files are located recursively, so it does not matter
whether the tar unpacks the shards at the top level of the dataset dir or in a
subfolder such as ``collections/``.

Example
-------
    # you already downloaded/extracted into ./dataset
    python prep_msmarco_v1.py --dataset-dir ../dataset

    # let the script download + extract the tar for you
    python prep_msmarco_v1.py --dataset-dir ../dataset --download

Memory note
-----------
The MS MARCO V1 passage corpus is ~8.8M vectors x 1536 dims (float32), i.e.
~54 GB just for the array, plus a similar amount inside the brute-force index.
Run this on a machine with plenty of RAM.
"""
import argparse
import glob
import gzip
import json
import os
import sys
import tarfile
import urllib.request

import numpy as np
import h5py
from tqdm import tqdm
import hnswlib

TAR_URL = "https://rgw.cs.uwaterloo.ca/pyserini/data/msmarco-passage-openai-ada2.tar"
TAR_NAME = "msmarco-passage-openai-ada2.tar"
QUERY_NAME = "topics.msmarco-passage.dev-subset.openai-ada2.jsonl.gz"
N_SHARDS = 89  # data shards named 0.jsonl.gz .. 88.jsonl.gz
K = 1000
VEC_FIELD = "vector"


def default_out_dir():
    root = os.environ.get("ADA_EF_ROOT")
    if root:
        return os.path.join(root, "experiments", "data")
    # fall back to the repo layout: experiments_driver/ -> ../experiments/data
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "experiments", "data"))


def find_one(root, name):
    """Return the single path to ``name`` anywhere under ``root``."""
    matches = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{name}' under '{root}'. "
            f"Did the download/extract finish? Try --download."
        )
    if len(matches) > 1:
        print(f"[warn] multiple matches for {name}, using {matches[0]}", file=sys.stderr)
    return matches[0]


def find_shards(root):
    paths = []
    for i in range(N_SHARDS):
        paths.append(find_one(root, f"{i}.jsonl.gz"))
    return paths


def _load_one(path):
    """Parse a single gzipped jsonl shard into a float32 array (worker fn).

    Each line is converted to a small float32 row immediately, so the transient
    Python-object list (1536 float objects) exists only for one line at a time.
    Accumulating raw parsed lists for the whole shard would cost ~5 GB of object
    overhead per shard; per-row float32 keeps the peak near the array size
    (~0.6 GB/shard).
    """
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rows.append(np.asarray(json.loads(line)[VEC_FIELD], dtype=np.float32))
    return np.vstack(rows)


def load_vectors(paths, desc, workers=1):
    # The shards are independent, and per-line JSON parsing is single-threaded
    # (GIL-bound), so we fan out across processes: each worker parses one shard
    # to float32 and we vstack the results in order. Keeping the corpus as one
    # big Python list of floats would need hundreds of GB of object overhead;
    # per-shard float32 arrays keep peak memory near the final array size.
    if workers > 1 and len(paths) > 1:
        import multiprocessing as mp
        # Cap workers to bound peak memory: each in-flight shard holds a ~1.2 GB
        # Python list plus a ~0.6 GB float32 array before it is consumed.
        n = min(workers, len(paths))
        with mp.Pool(processes=n) as pool:
            shard_arrays = list(
                tqdm(pool.imap(_load_one, paths), total=len(paths), desc=desc)
            )
    else:
        shard_arrays = [_load_one(p) for p in tqdm(paths, desc=desc)]
    if len(shard_arrays) == 1:
        return shard_arrays[0]
    return np.vstack(shard_arrays)


def download_and_extract(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    tar_path = os.path.join(dataset_dir, TAR_NAME)
    if not os.path.exists(tar_path):
        print(f"[info] downloading {TAR_URL}")
        urllib.request.urlretrieve(TAR_URL, tar_path)
    else:
        print(f"[info] tar already present: {tar_path}")
    print(f"[info] extracting {tar_path} -> {dataset_dir}")
    with tarfile.open(tar_path) as tar:
        tar.extractall(dataset_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default="dataset",
                    help="directory containing (or to receive) the MS MARCO V1 files")
    ap.add_argument("--download", action="store_true",
                    help="download + extract the tar before processing")
    ap.add_argument("--out-dir", default=None,
                    help="output dir for msmarco.hdf5 (default: $ADA_EF_ROOT/experiments/data)")
    ap.add_argument("--threads", type=int, default=-1,
                    help="threads for hnswlib brute-force GT (-1 = all cores)")
    ap.add_argument("--load-workers", type=int, default=24,
                    help="parallel processes for parsing the passage shards "
                         "(each needs ~1-2 GB RAM; too many will OOM)")
    args = ap.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else default_out_dir()
    os.makedirs(out_dir, exist_ok=True)

    if args.download:
        download_and_extract(dataset_dir)

    print(f"[info] locating files under {dataset_dir}")
    shard_paths = find_shards(dataset_dir)
    query_path = find_one(dataset_dir, QUERY_NAME)

    print(f"[info] loading {len(shard_paths)} shards with {args.load_workers} workers")
    data_vecs = load_vectors(shard_paths, "passages", workers=args.load_workers)
    print(f"[info] passage vectors: {data_vecs.shape}")
    query_vecs = load_vectors([query_path], "queries")
    print(f"[info] query vectors:   {query_vecs.shape}")

    print(f"[info] computing k={K} ground truth (brute-force cosine)")
    bf = hnswlib.BFIndex(space="cosine", dim=data_vecs.shape[1])
    bf.init_index(max_elements=data_vecs.shape[0])
    if args.threads != -1:
        bf.set_num_threads(args.threads)
    bf.add_items(data_vecs)
    gt, _ = bf.knn_query(query_vecs, k=K)

    out_path = os.path.join(out_dir, "msmarco.hdf5")
    print(f"[info] writing {out_path}")
    with h5py.File(out_path, "w") as h5f:
        h5f.create_dataset("test", data=query_vecs)
        h5f.create_dataset("train", data=data_vecs)
        h5f.create_dataset("neighbors", data=gt)

    print(f"[done] Saved MS MARCO V1 dataset and ground truth to {out_path}")


if __name__ == "__main__":
    main()
