"""Read-only performance/HTTP review of an explicitly specified large dataset.

Keeps the warmed application available for browser UX checks with --serve.
No index generation or metadata writes are performed.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import os
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def process_memory():
    if os.name != 'nt':
        return {}
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]
    info = Counters()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(info), info.cb):
        raise ctypes.WinError()
    return {'working_set_bytes': info.WorkingSetSize, 'private_bytes': info.PrivateUsage, 'peak_working_set_bytes': info.PeakWorkingSetSize}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--meta-dir', type=Path, required=True)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--output', type=Path, default=ROOT / '.cache/large-review.json')
    args = parser.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location('entry', ROOT / 'app.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    from app.application.usecase import Usecase
    from app.infrastructure.local_accessor import LocalAccessor
    from app.infrastructure.local_repository import LocalRepository
    from app.presentation import controller

    logging.basicConfig(level=logging.WARNING)
    report = {'dataset': 'explicit dataset paths', 'process_id': os.getpid(), 'operations': {}}

    def measure(label, function):
        print(f'START {label}', flush=True)
        started = perf_counter()
        result = function()
        elapsed = round((perf_counter() - started) * 1000, 2)
        report['operations'][label] = {'ms': elapsed}
        report['operations'][label].update(process_memory())
        print(f'END {label}: {elapsed} ms', flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        return result

    accessor = LocalAccessor(args.meta_dir)
    repository = LocalRepository(args.image_dir, args.meta_dir)
    model = entry.select_startup_model(repository.load_all_model_item())
    usecase = measure('catalog_startup', lambda: Usecase(repository, accessor, model))
    measure('warmup', lambda: usecase.warmup_search_cache(model))
    measure('catalog_count', lambda: usecase.get_catalog_count(model))
    measure('catalog_category_counts', lambda: usecase.get_catalog_count(model, ['general']))
    controller.usecase = usecase
    client = controller.app.test_client()
    options = {'model_name': model.model_name, 'pretrained': model.pretrained, 'result_size': 2048}

    def request(label, method, route, payload=None):
        response = measure(label, lambda: client.open(route, method=method, json=payload))
        assert response.status_code == 200, (label, response.status_code, response.data[:200])
        value = response.get_json() if response.is_json else None
        report['operations'][label].update(status=response.status_code, bytes=len(response.data))
        return value

    first = request('first_page', 'GET', '/image_item?page=0&size=60')
    request('later_page', 'GET', '/image_item?page=100&size=60')
    request('thumbnail_cold', 'GET', f"/image/{first[0]['id']}/small")
    request('thumbnail_cached', 'GET', f"/image/{first[0]['id']}/small")
    text = request('text_search_first', 'POST', '/search/text', {'params': {**options, 'text': 'a cat'}})
    request('text_search_repeat', 'POST', '/search/text', {'params': {**options, 'text': 'a cat'}})
    replay = request('saved_query', 'POST', '/search/query', {'params': {**options, 'search_query': text['search_query']}})
    assert text['list'] == replay['list']
    report['search_results'] = len(text['list'])
    report['index_rows'] = accessor.load_index_with_metadata(model, 'original')[0].ntotal
    report['saved_query_matches'] = True
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if args.serve:
        print(f'READY http://127.0.0.1:{args.port}', flush=True)
        controller.app.run(host='127.0.0.1', port=args.port, debug=False)


if __name__ == '__main__':
    main()
