import io
import unittest
from unittest.mock import Mock, patch

from app.presentation import controller
from app.presentation.download_store import DownloadStore, DownloadBusyError


class PreparedDownloadTests(unittest.TestCase):
    def setUp(self):
        self.store = DownloadStore(ttl_seconds=300, max_pending=1)
        self.usecase = Mock()
        self.stream = io.BytesIO(b'ZIP payload')
        self.stream.requested_count = 3
        self.stream.image_count = 2
        self.stream.skipped_count = 1
        self.usecase.get_images_zip.return_value = self.stream
        self.usecase.get_download_ids.return_value = ['first', 'second', 'missing']
        self.patch_store = patch.object(controller, 'download_store', self.store)
        self.patch_usecase = patch.object(controller, 'usecase', self.usecase)
        self.patch_store.start()
        self.patch_usecase.start()
        self.addCleanup(self.patch_store.stop)
        self.addCleanup(self.patch_usecase.stop)
        self.addCleanup(lambda: [self.store.expire(token) for token in list(self.store._entries)])
        self.client = controller.app.test_client()

    def prepare(self, fields=None):
        return self.client.post('/downloads/prepare', json={'params': fields or {'ids': ['a', 'b', 'missing']}})

    def test_browser_download_is_one_use_streamed_and_reports_missing_images(self):
        response = self.prepare()
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual((data['requested_count'], data['image_count'], data['skipped_count']), (3, 2, 1))
        self.assertEqual(data['bytes'], 11)
        download = self.client.get(data['download_url'])
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.data, b'ZIP payload')
        self.assertEqual(download.headers['Cache-Control'], 'no-store')
        self.assertEqual(download.content_length, 11)
        self.assertIn('attachment;', download.headers['Content-Disposition'])
        self.assertEqual(self.client.get(data['download_url']).status_code, 410)
        self.assertEqual(self.prepare().status_code, 429)
        download.close()
        self.assertTrue(self.stream.closed)
        self.usecase.get_images_zip.return_value = io.BytesIO(b'new')
        self.assertEqual(self.prepare().status_code, 200)

    def test_expired_preparations_close_the_file_and_release_capacity(self):
        data = self.prepare().get_json()
        self.store.expire(data['download_url'].split('/')[-1])
        self.assertTrue(self.stream.closed)
        self.assertEqual(self.client.get(data['download_url']).status_code, 410)
        self.usecase.get_images_zip.return_value = io.BytesIO(b'new')
        self.assertEqual(self.prepare().status_code, 200)

    def test_failed_creation_releases_capacity(self):
        with self.assertRaises(ValueError):
            self.store.prepare(lambda: (_ for _ in ()).throw(ValueError('failed')))
        self.assertEqual(self.prepare().status_code, 200)

    def test_busy_download_does_not_resolve_large_catalog_candidates(self):
        self.assertEqual(self.prepare().status_code, 200)
        response = self.prepare({'first': 1024, 'model_name': 'other', 'ratings': ['general']})
        self.assertEqual(response.status_code, 429)
        self.usecase.get_download_ids.assert_not_called()

    def test_first_mode_uses_the_same_validated_selection_as_legacy_post(self):
        response = self.prepare({'first': 1024, 'model_name': 'other', 'pretrained': '', 'ratings': ['general']})
        self.assertEqual(response.status_code, 200)
        self.usecase.get_images_zip.assert_called_once_with(['first', 'second', 'missing'])
        self.assertEqual(self.usecase.get_download_ids.call_args.args[1:], (1024, ['general']))

    def test_send_failure_closes_stream_and_releases_capacity(self):
        data = self.prepare().get_json()
        with patch.object(controller, 'send_file', side_effect=OSError('send failed')):
            with self.assertLogs(controller.app.logger, level='ERROR'):
                self.assertEqual(self.client.get(data['download_url']).status_code, 500)
        self.assertTrue(self.stream.closed)


if __name__ == '__main__':
    unittest.main()
