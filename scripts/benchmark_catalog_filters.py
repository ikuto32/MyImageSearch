"""Measure catalog counts and page retrieval for every image category, read-only."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta-dir', type=Path, default=ROOT / 'clip_meta')
    parser.add_argument('--model-name', default='ViT-L-14')
    parser.add_argument('--pretrained', default='openai')
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/catalog_filter_benchmark.json')
    args = parser.parse_args()

    from app.domain.domain_object import ModelId
    from app.domain.rating import RATING_CATEGORIES
    from app.infrastructure.catalog_database import CatalogDatabase
    from app.infrastructure.model_registry import ModelRegistry

    catalog = CatalogDatabase(ModelRegistry(args.meta_dir))
    model = ModelId(args.model_name, args.pretrained)
    report = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def measure(name, operation):
        start = time.perf_counter()
        value = operation()
        row = {'operation': name, 'seconds': round(time.perf_counter() - start, 6), 'result': value}
        report.append(row)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return value

    def page(number, ratings=None):
        return lambda: len(catalog.page(model, number, 60, ratings))

    count = measure('count_all_first', lambda: catalog.count(model))
    measure('page_all_last', page((count - 1) // 60))
    measure('page_all_middle', page(count // 120))
    for rating in RATING_CATEGORIES:
        ratings = (rating,)
        total = measure(f'count_{rating}', lambda: catalog.count(model, ratings))
        measure(f'count_{rating}_cached', lambda: catalog.count(model, ratings))
        measure(f'page_{rating}_first', page(0, ratings))
        if total:
            middle = total // 120
            measure(f'page_{rating}_last', page((total - 1) // 60, ratings))
            measure(f'page_{rating}_middle', page(middle, ratings))
            measure(f'page_{rating}_middle_next', page(middle + 1, ratings))
            measure(f'page_{rating}_middle_previous', page(max(0, middle - 1), ratings))


if __name__ == '__main__':
    main()
