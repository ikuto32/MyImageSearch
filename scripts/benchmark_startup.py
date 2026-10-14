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
    from app.infrastructure.model_registry import ModelRegistry

    model = ModelId(args.model_name, args.pretrained)
    tracemalloc.start()
    started = perf_counter()
    registry = ModelRegistry(args.meta_dir)
    accessor = LocalAccessor(registry)
    # No image method is called. Using the bundled root avoids contacting a NAS.
    usecase = Usecase(LocalRepository(ROOT / 'images', registry), accessor)
    first = usecase.get_image_items_by_page(model, 0, 60)
    startup_ms = (perf_counter() - started) * 1000
    current, peak = tracemalloc.get_traced_memory()
    started = perf_counter()
    second = usecase.get_image_items_by_page(model, 1, 60)
    page_ms = (perf_counter() - started) * 1000
    tracemalloc.stop()
    assert len(accessor._search_indexes) == 0, 'Catalog browsing unexpectedly loaded FAISS.'
    print(json.dumps({
        'models': len(registry.items()),
        'startup_and_first_page_ms': round(startup_ms, 2), 'first_page_rows': len(first),
        'python_retained_bytes': current, 'python_peak_bytes': peak,
        'second_page_ms': round(page_ms, 2), 'second_page_rows': len(second),
    }, indent=2))


if __name__ == '__main__':
    main()
