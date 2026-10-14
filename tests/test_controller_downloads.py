import gc
import io
import unittest
from unittest.mock import Mock, patch

from app.application.usecase import ImageArchive
from app.domain import limits
from app.domain.domain_object import ModelId
from app.domain.errors import ResourceLimitError, SearchInputError
from app.presentation import controller
from app.presentation.controller import create_app
from app.presentation.download_store import DownloadBusyError, DownloadStore


class PreparedDownloadTests(unittest.TestCase):
    def setUp(self):
        self.store = DownloadStore(ttl_seconds=300, max_pending=1)
        self.addCleanup(self.store.expire_all)
        self.usecase = Mock()
        self.stream = io.BytesIO(b"ZIP payload")
        self.usecase.get_images_zip.return_value = ImageArchive(self.stream, 3, 2, 1)
        self.usecase.get_download_ids.return_value = ["first", "second", "missing"]
        self.app = create_app(self.usecase, self.store)
        self.client = self.app.test_client()

    def prepare(self, fields=None):
        return self.client.post("/downloads/prepare", json={"params": fields or {"ids": ["a", "b", "missing"]}})

    def another_archive(self):
        self.usecase.get_images_zip.return_value = ImageArchive(io.BytesIO(b"new"), 1, 1, 0)

    def test_browser_download_is_one_use_streamed_and_reports_missing_images(self):
        response = self.prepare()
        self.assertEqual(response.status_code, 200)
        self.usecase.get_images_zip.assert_called_once_with(["a", "b", "missing"])
        data = response.get_json()
        self.assertEqual((data["requested_count"], data["image_count"], data["skipped_count"]), (3, 2, 1))
        self.assertEqual(data["bytes"], 11)
        download = self.client.get(data["download_url"])
        self.assertEqual(download.status_code, 200)
        self.assertEqual(download.data, b"ZIP payload")
        self.assertEqual(download.headers["Cache-Control"], "no-store")
        self.assertEqual(download.content_length, 11)
        self.assertIn("attachment;", download.headers["Content-Disposition"])
        expired = self.client.get(data["download_url"])
        self.assertEqual((expired.status_code, expired.get_json()["error_code"]), (410, "download_expired"))
        busy = self.prepare()
        self.assertEqual((busy.status_code, busy.get_json()["error_code"]), (429, "download_busy"))
        download.close()
        self.assertTrue(self.stream.closed)
        self.another_archive()
        self.assertEqual(self.prepare().status_code, 200)

    def test_head_request_does_not_use_up_the_download(self):
        data = self.prepare().get_json()
        probe = self.client.head(data["download_url"])
        self.assertEqual((probe.status_code, probe.headers["Allow"]), (405, "GET"))
        self.assertFalse(self.stream.closed)
        download = self.client.get(data["download_url"])
        self.addCleanup(download.close)
        self.assertEqual((download.status_code, download.data), (200, b"ZIP payload"))

    def test_expired_preparations_close_the_file_and_release_capacity(self):
        data = self.prepare().get_json()
        self.store.expire(data["download_url"].split("/")[-1])
        self.assertTrue(self.stream.closed)
        self.assertEqual(self.client.get(data["download_url"]).status_code, 410)
        self.another_archive()
        self.assertEqual(self.prepare().status_code, 200)

    def test_failed_creation_releases_capacity(self):
        for error, status in ((SearchInputError("保存できる画像がありません。"), 400), (ResourceLimitError("too large"), 413)):
            self.usecase.get_images_zip.side_effect = error
            response = self.prepare()
            self.assertEqual((response.status_code, response.get_json()["error"]), (status, str(error)))
        self.usecase.get_images_zip.side_effect = None
        self.assertEqual(self.prepare().status_code, 200)

    def test_busy_download_does_not_resolve_large_catalog_candidates(self):
        self.assertEqual(self.prepare().status_code, 200)
        response = self.prepare({"first": 1024, "model_name": "other", "ratings": ["general"]})
        self.assertEqual(response.status_code, 429)
        self.usecase.get_download_ids.assert_not_called()

    def test_first_mode_archives_the_filtered_head_of_the_selected_catalog(self):
        response = self.prepare({"first": 1024, "model_name": "other", "pretrained": "", "ratings": ["general"]})
        self.assertEqual(response.status_code, 200)
        self.usecase.get_download_ids.assert_called_once_with(ModelId("other", ""), 1024, ["general"])
        self.usecase.get_images_zip.assert_called_once_with(["first", "second", "missing"])

    def test_selection_size_and_mode_are_validated_before_archive_creation(self):
        accepted = self.prepare({"ids": [str(i) for i in range(limits.MAX_DOWNLOAD_IMAGES)]})
        self.assertEqual(accepted.status_code, 200)
        self.store.expire_all()
        self.usecase.reset_mock()
        for params in (
            {"ids": [str(i) for i in range(limits.MAX_DOWNLOAD_IMAGES + 1)]},
            {"ids": []}, {"ids": "a"}, {"ids": [1]},
            {"first": limits.MAX_DOWNLOAD_IMAGES + 1, "model_name": "Main"},
            {"first": 0, "model_name": "Main"},
            {"first": 1, "ids": ["a"], "model_name": "Main"},
            {"first": 1},
            {"first": 1, "model_name": "Main", "ratings": ["unknown"]},
            {"first": 1, "model_name": "Main", "ratings": "general"},
        ):
            with self.subTest(params=params):
                self.assertEqual(self.prepare(params).status_code, 400)
        self.assertEqual(self.usecase.mock_calls, [])
        # A rejected request must not keep occupying the only download slot.
        self.another_archive()
        self.assertEqual(self.prepare().status_code, 200)

    def test_send_failure_closes_stream_and_releases_capacity(self):
        data = self.prepare().get_json()
        with patch.object(controller, "send_file", side_effect=OSError("send failed")):
            with self.assertLogs(self.app.logger, level="ERROR"):
                self.assertEqual(self.client.get(data["download_url"]).status_code, 500)
        self.assertTrue(self.stream.closed)
        self.another_archive()
        self.assertEqual(self.prepare().status_code, 200)


class DownloadStoreTests(unittest.TestCase):
    def test_capacity_is_released_when_building_fails(self):
        store = DownloadStore(max_pending=1)
        self.addCleanup(store.expire_all)
        with self.assertRaises(ValueError):
            store.prepare(lambda: (_ for _ in ()).throw(ValueError("failed")))
        token, size = store.prepare(lambda: io.BytesIO(b"12345"))
        self.assertEqual(size, 5)
        with self.assertRaises(DownloadBusyError):
            store.prepare(lambda: io.BytesIO())
        stream = store.consume(token)
        self.assertEqual(stream.read(), b"12345")
        self.assertIsNone(store.consume(token))
        stream.close()
        stream.close()
        store.prepare(lambda: io.BytesIO())

    def test_capacity_is_released_when_a_consumed_stream_is_dropped_without_closing(self):
        # A server whose client disconnects mid-transfer may never close the response.
        store = DownloadStore(max_pending=1)
        self.addCleanup(store.expire_all)
        source = io.BytesIO(b"12345")
        token, _size = store.prepare(lambda: source)
        stream = store.consume(token)
        self.assertEqual(stream.read(2), b"12")
        del stream
        gc.collect()
        self.assertTrue(source.closed)
        store.prepare(lambda: io.BytesIO())

    def test_stale_entries_cannot_be_consumed(self):
        store = DownloadStore(ttl_seconds=0, max_pending=1)
        self.addCleanup(store.expire_all)
        source = io.BytesIO(b"old")
        with patch("app.presentation.download_store.threading.Timer"):
            token, _size = store.prepare(lambda: source)
        self.assertIsNone(store.consume(token))
        self.assertTrue(source.closed)
        store.prepare(lambda: io.BytesIO())


if __name__ == "__main__":
    unittest.main()
