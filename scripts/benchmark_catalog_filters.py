import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser(description='Benchmark catalog counts and page retrieval.')
parser.add_argument('--meta-dir', type=Path, default=ROOT / 'clip_meta')
args = parser.parse_args()

from app.domain.domain_object import ModelId
from app.infrastructure.local_accessor import LocalAccessor

accessor = LocalAccessor(args.meta_dir)
model = ModelId('ViT-L-14', 'openai')
report = []
out = ROOT / '.cache/catalog_filter_benchmark.json'
out.parent.mkdir(parents=True, exist_ok=True)

def measure(name, operation):
    start = time.perf_counter()
    value = operation()
    row = {'operation':name, 'seconds':round(time.perf_counter()-start, 6), 'result':value}
    report.append(row)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(row,ensure_ascii=False),flush=True)
    return value

count = measure('count_all_first', lambda: accessor.load_catalog_count(model))
measure('page_all_last', lambda: len(accessor.load_startup_image_page(model,(count-1)//60,60)[0]))
measure('page_all_middle', lambda: len(accessor.load_startup_image_page(model,count//120,60)[0]))
for rating in ['general','sensitive','questionable','explicit','unclassified']:
    total = measure('count_'+rating, lambda: accessor.load_catalog_count(model,[rating]))
    measure('count_'+rating+'_cached', lambda: accessor.load_catalog_count(model,[rating]))
    measure('page_'+rating+'_first', lambda: len(accessor.load_startup_image_page(model,0,60,[rating])[0]))
    if total:
        measure('page_'+rating+'_last', lambda: len(accessor.load_startup_image_page(model,(total-1)//60,60,[rating])[0]))
        measure('page_'+rating+'_middle', lambda: len(accessor.load_startup_image_page(model,total//120,60,[rating])[0]))
        measure('page_'+rating+'_middle_next', lambda: len(accessor.load_startup_image_page(model,total//120+1,60,[rating])[0]))
        measure('page_'+rating+'_middle_previous', lambda: len(accessor.load_startup_image_page(model,total//120-1,60,[rating])[0]))
