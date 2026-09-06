"""Measure read-only catalog startup/paging without loading images or ML models.

Python allocation figures exclude imports, SQLite/native allocations and OS caches.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta-dir', type=Path, default=ROOT / 'clip_meta')
    parser.add_argument('--model-name', default='ViT-L-14')
    parser.add_argument('--pretrained', default='openai')
    args = parser.parse_args()

    from app.application.usecase import Usecase
    from app.domain.domain_object import ModelId
    from app.infrastructure.local_accessor import LocalAccessor
    from app.infrastructure.local_repository import LocalRepository

    accessor = LocalAccessor(args.meta_dir)
    # No image method is called. Using the bundled root avoids contacting a NAS.
    repository = LocalRepository(ROOT / 'images', args.meta_dir)
    tracemalloc.start()
    started = perf_counter()
    usecase = Usecase(repository, accessor, ModelId(args.model_name, args.pretrained))
    startup_ms = (perf_counter() - started) * 1000
    current, peak = tracemalloc.get_traced_memory()
    first_count = len(usecase._id_to_image_items)
    assert usecase._paged_catalog, 'This benchmark requires a registered catalog database.'
    started = perf_counter()
    second = usecase.get_image_items_by_page(1, 60)
    page_ms = (perf_counter() - started) * 1000
    tracemalloc.stop()
    assert not accessor._search_indexes, 'Catalog browsing unexpectedly loaded FAISS.'
    print(json.dumps({
        'startup_ms': round(startup_ms, 2), 'startup_rows': first_count,
        'startup_python_retained_bytes': current, 'startup_python_peak_bytes': peak,
        'second_page_ms': round(page_ms, 2), 'second_page_rows': len(second),
        'retained_rows_after_second_page': len(usecase._id_to_image_items),
        'loaded_faiss_indexes': len(accessor._search_indexes),
    }, indent=2))


if __name__ == '__main__':
    main()
