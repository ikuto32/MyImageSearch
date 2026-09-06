r"""Measure only metadata lookup overhead before and after removing per-search sort.

Run from the project root:
    .\.venv\Scripts\python.exe scripts/benchmark_metadata.py

The synthetic metadata is already ordered by path, as LocalAccessor returns it.
The old path sorts all records again before looking up the FAISS result positions;
the new path uses those positions directly. No FAISS search, embedding model,
database, images, network, or result scoring is measured here.

Timing runs do not enable tracemalloc. A separate allocation run starts tracing
after fixture construction, so peaks exclude the existing metadata and indices.
"""

import argparse
import gc
import json
import platform
import random
import statistics
import time
import tracemalloc
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageName:
    name: str


@dataclass(frozen=True, slots=True)
class ImageItem:
    display_name: ImageName


def old_metadata_lookup(items, matched_indices):
    sorted_items = sorted(items, key=lambda item: item.display_name.name)
    return [sorted_items[index] for index in matched_indices]


def direct_metadata_lookup(items, matched_indices):
    return [items[index] for index in matched_indices]


def measure_lookup(lookup, items, matched_indices, repeats):
    samples_ms = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        result = lookup(items, matched_indices)
        samples_ms.append((time.perf_counter() - started) * 1000)
        del result

    gc.collect()
    tracemalloc.start()
    try:
        result = lookup(items, matched_indices)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del result

    return {
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
        "extra_peak_bytes": peak_bytes,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=1_000_000)
    parser.add_argument("--results", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if args.items < 1 or not 1 <= args.results <= args.items or args.repeats < 1:
        parser.error("items/repeats must be positive and results must be between 1 and items")

    # Fixed-width names preserve lexical path order for every supported item count.
    width = len(str(args.items))
    items = [ImageItem(ImageName(f"image_{index:0{width}d}.jpg")) for index in range(args.items)]
    matched_indices = random.Random(20260906).sample(range(args.items), args.results)

    old_result = old_metadata_lookup(items, matched_indices)
    new_result = direct_metadata_lookup(items, matched_indices)
    if any(old is not new for old, new in zip(old_result, new_result)):
        raise AssertionError("The old and new paths returned different image references")
    del old_result, new_result

    old = measure_lookup(old_metadata_lookup, items, matched_indices, args.repeats)
    direct = measure_lookup(direct_metadata_lookup, items, matched_indices, args.repeats)
    report = {
        "scope": "Metadata reference selection only; not end-to-end search latency",
        "metadata_order": "Already sorted by path, matching LocalAccessor",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "item_count": args.items,
        "result_count": args.results,
        "repeats": args.repeats,
        "same_image_references": True,
        "old_sort_then_lookup": old,
        "direct_lookup": direct,
        "median_speedup": old["median_ms"] / direct["median_ms"],
        "extra_peak_reduction_bytes": old["extra_peak_bytes"] - direct["extra_peak_bytes"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
