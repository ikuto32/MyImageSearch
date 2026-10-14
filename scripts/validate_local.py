"""Read-only smoke checks against a prepared local images/clip_meta dataset.

Run from any directory: python scripts/validate_local.py
No embedding weights are loaded unless --with-text is supplied.
"""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-dir', type=Path, default=ROOT / 'images')
    parser.add_argument('--meta-dir', type=Path, default=ROOT / 'clip_meta')
    parser.add_argument('--with-text', action='store_true')
    parser.add_argument('--all-models', action='store_true', help='全登録モデルでindexとクエリ再実行を検証（推論なし）')
    args = parser.parse_args()

    from app.application.usecase import Usecase
    from app.domain.domain_object import ModelId
    from app.infrastructure.local_accessor import LocalAccessor
    from app.infrastructure.local_repository import LocalRepository
    from app.infrastructure.model_registry import ModelRegistry
    from app.presentation.controller import create_app

    model = ModelId('ViT-L-14', 'openai')
    start = perf_counter()
    registry = ModelRegistry(args.meta_dir)
    accessor = LocalAccessor(registry)
    usecase = Usecase(LocalRepository(args.image_dir, registry), accessor)
    client = create_app(usecase).test_client()
    report = {'startup_ms': round((perf_counter() - start) * 1000, 2)}
    options = {'model_name': model.model_name, 'pretrained': model.pretrained, 'result_size': 12}

    def post(route, fields):
        start = perf_counter()
        response = client.post(route, json={'params': {**options, **fields}})
        assert response.status_code == 200, (route, response.status_code, response.data[:300])
        assert response.mimetype == 'application/json', (route, response.content_type)
        assert response.headers.get('X-Content-Type-Options') == 'nosniff'
        payload = response.get_json()
        assert len(payload['list']) <= options['result_size']
        report[route] = {'ms': round((perf_counter() - start) * 1000, 2), 'results': len(payload['list'])}
        return payload

    def assert_replay(original):
        replay = post('/search/query', {'search_query': original['search_query']})
        assert original['list'] == replay['list'], '保存クエリの順位またはスコアが変化しました'

    catalog = {'model_name': model.model_name, 'pretrained': model.pretrained}
    page_response = client.get('/image_item', query_string={**catalog, 'size': 12, 'include_total': 1})
    page = page_response.get_json(force=True)
    assert len(page) == 12
    report['catalog_total'] = int(page_response.headers['X-Catalog-Total'])
    first = page[0]
    thumbnail = client.get(f"/image/{first['id']}/small")
    assert thumbnail.status_code == 200 and thumbnail.content_type.startswith('image/')
    report['thumbnail_bytes'] = len(thumbnail.data)
    metadata = client.get(f"/image_meta/{first['id']}", query_string=catalog)
    assert metadata.status_code == 200 and metadata.get_json()['rating'] == first['rating']
    names = post('/search/name', {'text': first['name']})
    assert first['id'] in [result['item']['id'] for result in names['list']]
    post('/search/tags', {'text': 'cat'})
    random = post('/search/random', {})
    assert random['list']
    assert_replay(random)
    report['images'] = len(accessor.load_search_index(model))
    assert client.post('/search/name', json={'params': {**options, 'text': '[', 'is_regexp': True}}).status_code == 400
    assert client.get('/image_item', query_string={**catalog, 'size': 241}).status_code == 400
    assert client.get('/image/unknown-image/small').status_code == 404
    if args.with_text:
        text_result = post('/search/text', {'text': 'a cat'})
        assert text_result['list']
        assert_replay(text_result)
    if args.all_models:
        models = []
        for model_item in registry.items():
            model_id = model_item.id
            options.update(model_name=model_id.model_name, pretrained=model_id.pretrained)
            rows = len(accessor.load_search_index(model_id))
            result = post('/search/random', {})
            assert result['list']
            assert_replay(result)
            image_id = result['list'][0]['item']['id']
            assert client.get(f'/image/{image_id}/small').status_code == 200
            models.append({'name': model_id.model_name, 'pretrained': model_id.pretrained, 'rows': rows})
        report['models'] = models
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
