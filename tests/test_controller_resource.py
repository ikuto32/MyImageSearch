import unittest
from unittest.mock import Mock

from app.presentation.controller import VIEW_DIRECTORY, create_app


class ResourceRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app(Mock()).test_client()
        self.evil_dir = VIEW_DIRECTORY.parent / "view_evil"
        self.evil_file = self.evil_dir / "file.js"
        self.evil_dir.mkdir(exist_ok=True)
        self.evil_file.write_text("alert('evil');", encoding="UTF-8")

    def tearDown(self):
        self.evil_file.unlink(missing_ok=True)
        self.evil_dir.rmdir()

    def test_index_and_module_scripts_are_served_with_usable_types(self):
        index = self.client.get("/")
        self.assertEqual((index.status_code, index.mimetype), (200, "text/html"))
        self.assertIn(b'<div id="app"', index.data)
        script = self.client.get("/js/main.js")
        self.assertEqual((script.status_code, script.mimetype), (200, "text/javascript"))
        self.assertEqual(script.headers["X-Content-Type-Options"], "nosniff")
        language = self.client.get("/lang/ja.json")
        self.assertEqual(language.mimetype, "application/json")
        self.assertEqual(self.client.get("/js/missing.js").status_code, 404)
        for response in (index, script, language):
            response.close()

    def test_rejects_sibling_view_evil_path_traversal(self):
        response = self.client.get("/../view_evil/file.js")

        self.assertEqual(response.status_code, 404)
        self.assertNotIn(b"alert('evil')", response.data)

    def test_rejects_url_encoded_dot_dot_path_traversal(self):
        response = self.client.get("/js/%2e%2e/%2e%2e/view_evil/file.js")

        self.assertEqual(response.status_code, 404)
        self.assertNotIn(b"alert('evil')", response.data)


if __name__ == "__main__":
    unittest.main()
