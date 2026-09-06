from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from app.domain.domain_object import ModelId
from app.domain.errors import SearchInputError
from app.infrastructure.local_accessor import LocalAccessor
from app.infrastructure.model_metadata import open_readonly_database
from app.presentation import controller


class FilteredCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = ModelId('Main', 'weights')
        self.rows = [
            ('z', 'z.png', 'z', 'general'), ('b', 'a.png', 'b', None),
            ('a', 'a.png', 'a', 'general'), ('c', 'b.png', 'c', 'safe'),
            ('d', 'c.png', 'd', ''), ('e', 'd.png', 'e', 'explicit'),
        ]
        self.create_catalog('Main-weights', self.rows)
        self.accessor = LocalAccessor(self.root)

    def create_catalog(self, name, rows, rating=True):
        folder = self.root / name
        folder.mkdir()
        (folder / 'metafiles.index').write_bytes(b'index must not be loaded')
        with closing(sqlite3.connect(folder / 'sqlite_image_meta.db')) as con:
            column = ', rating TEXT' if rating else ''
            con.execute(f'CREATE TABLE image_meta (image_id TEXT PRIMARY KEY, image_path TEXT, image_tags TEXT{column})')
            con.execute('CREATE INDEX path_order ON image_meta(image_path, image_id)')
            marks = '?,?,?,?' if rating else '?,?,?'
            con.executemany(f'INSERT INTO image_meta VALUES ({marks})', rows)
            con.commit()

    def test_filtered_pages_count_and_zip_share_the_same_normalization_and_order(self):
        for ratings, expected in ((None, ['a','b','c','d','e','z']), ([], []), (['general'], ['a','z']), (['unclassified'], ['b','c','d']), (['explicit','general'], ['a','e','z'])):
            with self.subTest(ratings=ratings):
                count = self.accessor.load_catalog_count(self.model, ratings)
                actual = []
                for page in range((count + 1) // 2):
                    items, paths = self.accessor.load_startup_image_page(self.model, page, 2, ratings)
                    actual.extend(item.id.id for item in items)
                    self.assertEqual(len(items), len(paths))
                self.assertEqual(count, len(expected))
                self.assertEqual(actual, expected)
                self.assertEqual([item.id.id for item in self.accessor.load_download_image_items(self.model, 1024, ratings)], expected)
        self.assertEqual(self.accessor._search_indexes, {})

    def test_arbitrary_filtered_page_can_be_loaded_without_reading_prior_pages(self):
        rows = [(str(i), f'{i:05d}.png', 'tag', 'general' if i % 2 else 'explicit') for i in range(1000)]
        self.create_catalog('Other-weights', rows)
        other = ModelId('Other', 'weights')
        self.assertEqual(self.accessor.load_catalog_count(other, ['general']), 500)
        items, _ = self.accessor.load_startup_image_page(other, 8, 60, ['general'])
        self.assertEqual([item.id.id for item in items], [str(i) for i in range(961, 1000, 2)])
        self.assertTrue(all(item.rating == 'general' for item in items))
        self.assertEqual(self.accessor.load_catalog_count(other, ['general']), 500)

    def test_covering_path_seek_preserves_duplicate_path_and_null_boundary_order(self):
        rows = [('null-a', None, '', 'general'), ('null-b', None, '', 'general')]
        rows += [(f'a-{i}', f'a{i:04d}.png', '', 'general') for i in range(2000)]
        rows += [(f'dup-{1000-i:04d}', 'middle.png', '', 'general') for i in range(1000)]
        rows += [(f'z-{i}', f'z{i:04d}.png', '', 'general') for i in range(2000)]
        self.create_catalog('Large-weights', rows)
        model = ModelId('Large', 'weights')
        self.assertEqual(self.accessor.load_catalog_count(model), len(rows))
        with closing(open_readonly_database(self.root / 'Large-weights' / 'sqlite_image_meta.db')) as con:
            expected = [row[0] for row in con.execute('SELECT image_id FROM image_meta ORDER BY image_path, image_id LIMIT 60 OFFSET 2220')]
        items, _ = self.accessor.load_startup_image_page(model, 37, 60)
        self.assertEqual([item.id.id for item in items], expected)
        first, _ = self.accessor.load_startup_image_page(model, 0, 2)
        self.assertEqual([item.id.id for item in first], ['null-a','null-b'])
        self.assertEqual(self.accessor.load_startup_image_page(model, 1000, 60), ([], {}))

    def test_all_rating_combinations_share_one_sql_histogram_and_models_remain_independent(self):
        queries = []
        def connect(path):
            con = open_readonly_database(path)
            con.set_trace_callback(queries.append)
            return con
        with patch('app.infrastructure.local_accessor.open_readonly_database', side_effect=connect):
            self.assertEqual(self.accessor.load_catalog_count(self.model, ['general']), 2)
            self.assertEqual(self.accessor.load_catalog_count(self.model, ['explicit','general']), 3)
            self.assertEqual(self.accessor.load_catalog_count(self.model, ['general','explicit','general']), 3)
            self.assertEqual(self.accessor.load_catalog_count(self.model, ['unclassified']), 3)
            self.assertEqual(self.accessor.load_catalog_count(self.model), 6)
        aggregates = [q for q in queries if 'COUNT(*)' in q.upper()]
        self.assertEqual(len(aggregates), 1)
        self.create_catalog('Other-weights', [('other', 'other.png', '', 'general')])
        self.assertEqual(self.accessor.load_catalog_count(ModelId('Other', 'weights'), ['general']), 1)

    def test_adjacent_deep_filtered_pages_seek_from_a_recent_boundary(self):
        rows = [(str(i), f'{i:05d}.png', '', 'general') for i in range(12000)]
        self.create_catalog('Deep-weights', rows)
        model = ModelId('Deep', 'weights')
        self.assertEqual(self.accessor.load_catalog_count(model, ['general']), 12000)
        self.accessor.load_startup_image_page(model, 100, 60, ['general'])
        queries = []
        def connect(path):
            con = open_readonly_database(path)
            con.set_trace_callback(queries.append)
            return con
        with patch('app.infrastructure.local_accessor.open_readonly_database', side_effect=connect):
            items, _ = self.accessor.load_startup_image_page(model, 101, 60, ['general'])
        self.assertEqual([item.id.id for item in items], [str(i) for i in range(6060,6120)])
        page_queries = [q for q in queries if 'LIMIT' in q]
        self.assertEqual(len(page_queries), 1)
        self.assertIn('(image_path, image_id) >=', page_queries[0])
        self.assertIn('OFFSET 1', page_queries[0])
        self.assertNotIn('OFFSET 6060', page_queries[0])

    def test_concurrent_rating_changes_share_a_single_initial_aggregate(self):
        aggregates = []
        def connect(path):
            con = open_readonly_database(path)
            def trace(query):
                if 'GROUP BY' in query:
                    aggregates.append(query)
                    time.sleep(0.03)
            con.set_trace_callback(trace)
            return con
        ratings = [['general'], ['explicit'], ['unclassified'], ['sensitive'], ['general','explicit']]
        with patch('app.infrastructure.local_accessor.open_readonly_database', side_effect=connect):
            with ThreadPoolExecutor(max_workers=5) as pool:
                counts = list(pool.map(lambda value: self.accessor.load_catalog_count(self.model, value), ratings))
        self.assertEqual(counts, [2,1,3,0,3])
        self.assertEqual(len(aggregates), 1)

    def test_legacy_without_rating_is_only_unclassified(self):
        self.create_catalog('Legacy-weights', [('old', 'old.png', '')], rating=False)
        model = ModelId('Legacy', 'weights')
        self.assertEqual(self.accessor.load_catalog_count(model, ['unclassified']), 1)
        self.assertEqual(self.accessor.load_catalog_count(model, ['general']), 0)
        self.assertEqual(self.accessor.load_startup_image_page(model, 0, 60, ['general']), ([], {}))
        items, _ = self.accessor.load_startup_image_page(model, 0, 60, ['unclassified'])
        self.assertEqual(items[0].rating, '')

    def test_malformed_ratings_fail_without_any_catalog_scan(self):
        for ratings in ('general', [None], ['unknown'], [{}]):
            with self.subTest(ratings=ratings), patch('app.infrastructure.local_accessor.open_readonly_database', side_effect=AssertionError('must validate first')):
                with self.assertRaises(SearchInputError):
                    self.accessor.load_catalog_count(self.model, ratings)
                with self.assertRaises(SearchInputError):
                    self.accessor.load_startup_image_page(self.model, 0, 60, ratings)

    def test_api_exposes_global_and_filtered_totals_and_rejects_non_arrays(self):
        items, _ = self.accessor.load_startup_image_page(self.model, 0, 60, ['general'])
        usecase = Mock()
        usecase.get_image_items_by_page.return_value = items
        usecase.get_catalog_count.side_effect = lambda model, ratings=None: 6 if ratings is None else 2
        with patch.object(controller, 'usecase', usecase):
            client = controller.app.test_client()
            response = client.get('/image_item', query_string={'model_name':'Main', 'pretrained':'weights', 'page':0, 'size':60, 'ratings':json.dumps(['general']), 'include_total':1})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['X-Catalog-Total'], '6')
            self.assertEqual(response.headers['X-Matching-Total'], '2')
            self.assertEqual(response.get_json()[0]['rating'], 'general')
            usecase.get_image_items_by_page.assert_called_once_with(0, 60, self.model, ['general'])
            usecase.reset_mock()
            for value in ('general', '{}', 'null', '["unknown"]', '[null]', '[{}]'):
                self.assertEqual(client.get('/image_item', query_string={'ratings':value}).status_code, 400)
            usecase.get_image_items_by_page.assert_not_called()

    def test_known_empty_filter_returns_totals_without_a_redundant_page_scan(self):
        usecase = Mock()
        usecase.get_catalog_count.side_effect = lambda model, ratings=None: 7099334 if ratings is None else 0
        with patch.object(controller, 'usecase', usecase):
            response = controller.app.test_client().get('/image_item', query_string={'ratings':'["unclassified"]', 'include_total':1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), [])
        self.assertEqual(response.headers['X-Catalog-Total'], '7099334')
        self.assertEqual(response.headers['X-Matching-Total'], '0')
        usecase.get_image_items_by_page.assert_not_called()


if __name__ == '__main__':
    unittest.main()
