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
    parser.add_argument('--with-text', action='store_true')
    parser.add_argument('--all-models', action='store_true', help='全登録モデルでindexとクエリ再実行を検証（推論なし）')
    args = parser.parse_args()

    from app.application.usecase import Usecase
    from app.domain.domain_object import ModelId
    from app.infrastructure.local_accessor import LocalAccessor
    from app.infrastructure.local_repository import LocalRepository
    from app.presentation import controller

    model = ModelId('ViT-L-14', 'openai')
    accessor = LocalAccessor(ROOT / 'clip_meta')
    repository = LocalRepository(ROOT / 'images', ROOT / 'clip_meta')
    start = perf_counter()
    usecase = Usecase(repository, accessor, model)
    startup_ms = (perf_counter() - start) * 1000
    controller.usecase = usecase
    client = controller.app.test_client()
    report = {'startup_ms': round(startup_ms, 2), 'startup_retained_items': len(usecase._id_to_image_items)}
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

    page = client.get('/image_item?size=12').get_json(force=True)
    assert len(page) == 12
    first = page[0]
    thumbnail = client.get(f"/image/{first['id']}/small")
    assert thumbnail.status_code == 200 and thumbnail.content_type.startswith('image/')
    report['thumbnail_bytes'] = len(thumbnail.data)
    ratings = client.post('/image_ratings', json={'params': {**options, 'ids': [first['id']]}})
    assert ratings.status_code == 200
    names = post('/search/name', {'text': first['name']})
    assert first['id'] in [result['item']['id'] for result in names['list']]
    post('/search/tags', {'text': 'cat'})
    random = post('/search/random', {})
    assert random['list']
    assert_replay(random)
    index, _ = accessor.load_index_with_metadata(model, 'original')
    report['images'] = int(index.ntotal)
    assert accessor.load_index_with_metadata(model, 'pony')[0] is index
    assert client.post('/search/name', json={'params': {**options, 'text': '[', 'is_regexp': True}}).status_code == 400
    assert client.get('/image_item?size=241').status_code == 400
    assert client.post('/image_ratings', json={'params': options}).status_code == 400
    if args.with_text:
        text_result = post('/search/text', {'text': 'a cat'})
        assert text_result['list']
        assert_replay(text_result)
    if args.all_models:
        models = []
        for model_item in repository.load_all_model_item():
            model_id = model_item.id
            options.update(model_name=model_id.model_name, pretrained=model_id.pretrained)
            index, items = accessor.load_index_with_metadata(model_id, 'original')
            assert len(items) == index.ntotal
            result = post('/search/random', {})
            assert result['list']
            assert_replay(result)
            image_id = result['list'][0]['item']['id']
            assert client.get(f'/image/{image_id}/small').status_code == 200
            models.append({'name': model_id.model_name, 'pretrained': model_id.pretrained, 'rows': index.ntotal})
        report['models'] = models
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
