import base64
from io import BytesIO
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from app.application.search import SearchOptions
from app.application.text_matching import SearchPatternError
from app.domain import limits
from app.domain.domain_object import ImageId, ModelId, SearchResults
from app.domain.errors import ImageUnavailableError, ResourceLimitError, SearchInputError
from app.domain.search_query import InvalidSearchQuery, SearchQuery
from app.presentation import controller
from app.presentation.controller import create_app

MODEL = ModelId("ViT-B-32", "")
SEARCH_METHODS = (
    "search_text", "search_image", "search_upload_image", "search_random",
    "search_query", "add_text_features", "search_metadata",
)
SEARCH_ROUTES = (
    "/search/text", "/search/name", "/search/image", "/search/random",
    "/search/tags", "/search/style_cluster", "/search/query",
    "/search/queryaddtext", "/search/uploadimage",
)


class ControllerValidationTests(unittest.TestCase):
    def setUp(self):
        self.usecase = Mock()
        for method in SEARCH_METHODS:
            getattr(self.usecase, method).return_value = SearchResults([], "")
        self.app = create_app(self.usecase)
        self.client = self.app.test_client()
        self.saved_query = SearchQuery((0.1, -0.2, 1e-5), True, MODEL)

    def post(self, endpoint, **params):
        return self.client.post(endpoint, json={"params": {"model_name": "ViT-B-32", **params}})

    def assert_bad_request(self, response, parameter=None):
        self.assertEqual(response.status_code, 400, response.data)
        self.assertTrue(response.is_json)
        error = response.get_json().get("error")
        self.assertIsInstance(error, str)
        if parameter:
            self.assertIn(parameter, error)

    def test_all_post_routes_reject_bodies_without_a_params_object(self):
        bodies = ('{', 'null', '[]', '"text"', '{}', '{"params": null}', '{"params": []}', '{"model_name": "ViT-B-32"}')
        for endpoint in (*SEARCH_ROUTES, "/downloads/prepare"):
            for body in bodies:
                with self.subTest(endpoint=endpoint, body=body):
                    response = self.client.post(endpoint, data=body, content_type="application/json")
                    self.assert_bad_request(response)
        self.assertEqual(self.usecase.mock_calls, [])

    def test_deeply_nested_json_returns_400(self):
        body = '{"params":' + '[' * 2000 + '0' + ']' * 2000 + '}'
        self.assert_bad_request(self.client.post("/search/random", data=body, content_type="application/json"))

    def test_search_text_and_random_supply_shared_defaults(self):
        self.assertEqual(self.post("/search/text", text="cat").status_code, 200)
        self.usecase.search_text.assert_called_once_with(MODEL, "cat", SearchOptions())
        self.assertEqual(self.post("/search/random").status_code, 200)
        self.usecase.search_random.assert_called_once_with(MODEL, SearchOptions())
        self.assertEqual(SearchOptions(), SearchOptions(0.0, (0.0, 10.0), "original", 2048))

    def test_search_options_are_passed_to_every_search(self):
        expected = SearchOptions(-0.5, (2.0, 7.5), "pony", 60)
        options = {
            "aesthetic_quality_beta": -0.5, "aesthetic_quality_range": [2, 7.5],
            "aesthetic_model_name": "pony", "result_size": "60",
        }
        image = base64.b64encode(b"image bytes").decode("ascii")
        requests = (
            ("/search/text", {"text": "cat"}, "search_text"),
            ("/search/image", {"id": ["a"]}, "search_image"),
            ("/search/uploadimage", {"base64": image}, "search_upload_image"),
            ("/search/random", {}, "search_random"),
            ("/search/query", {"search_query": self.saved_query.to_text()}, "search_query"),
            ("/search/queryaddtext", {"text": "cat", "search_query": self.saved_query.to_text()}, "add_text_features"),
            ("/search/name", {"text": "cat"}, "search_metadata"),
        )
        for endpoint, fields, method in requests:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(self.post(endpoint, **fields, **options).status_code, 200)
                self.assertEqual(getattr(self.usecase, method).call_args.args[-1], expected)

    def test_null_optional_model_fields_use_their_defaults(self):
        response = self.post("/search/name", text="cat", is_regexp=True, pretrained=None, aesthetic_model_name=None)
        self.assertEqual(response.status_code, 200)
        self.usecase.search_metadata.assert_called_once_with(MODEL, "name", "cat", True, SearchOptions())

    def test_metadata_searches_select_their_field_and_accept_only_booleans(self):
        for field in ("name", "tags", "style_cluster"):
            for is_regexp in (True, False):
                with self.subTest(field=field, is_regexp=is_regexp):
                    self.assertEqual(self.post(f"/search/{field}", text="cat", is_regexp=is_regexp).status_code, 200)
                    self.assertEqual(self.usecase.search_metadata.call_args.args[1:4], (field, "cat", is_regexp))
        self.usecase.reset_mock()
        for value in (None, 0, 1, "yes", "true", [], {}):
            with self.subTest(value=value):
                self.assert_bad_request(self.post("/search/name", text="cat", is_regexp=value), "is_regexp")
        self.assert_bad_request(self.post("/search/name", text="[", is_regexp=True), "正規表現")
        self.assert_bad_request(self.post("/search/name", text="a{999999999999999999999}", is_regexp=True), "正規表現")
        self.usecase.search_metadata.assert_not_called()

    def test_model_and_text_types_are_validated_before_usecase(self):
        for parameter, invalid_values in (
            ("model_name", (None, "", "  ", [], 1)),
            ("text", (None, "", "  ", {}, 1)),
            ("pretrained", ([], True, 1)),
            ("aesthetic_model_name", ("", [], 1, "unknown")),
        ):
            for value in invalid_values:
                with self.subTest(parameter=parameter, value=value):
                    response = self.post("/search/text", **{"text": "cat", parameter: value})
                    self.assert_bad_request(response, parameter)
        self.usecase.search_text.assert_not_called()

    def test_invalid_ranges_and_nonfinite_numbers_are_rejected(self):
        cases = (
            ("aesthetic_quality_beta", (None, True, "0.1", -1.01, 1.01, float("nan"), float("inf"), 10 ** 400)),
            ("aesthetic_quality_range", (None, [], [0], [0, 1, 2], [5, 4], [-1, 5], [0, 11], [0, float("nan")], [False, 5])),
        )
        for parameter, values in cases:
            for value in values:
                with self.subTest(parameter=parameter, value=value):
                    self.assert_bad_request(self.post("/search/random", **{parameter: value}), parameter)
        self.usecase.search_random.assert_not_called()

    def test_result_size_rejects_invalid_values_and_accepts_limit(self):
        for value in (None, True, 0, -1, 1.5, "1.5", "invalid", limits.MAX_RESULT_SIZE + 1):
            with self.subTest(value=value):
                self.assert_bad_request(self.post("/search/random", result_size=value), "result_size")
        for value in (1, "60", limits.MAX_RESULT_SIZE):
            self.assertEqual(self.post("/search/random", result_size=value).status_code, 200)
            self.assertEqual(self.usecase.search_random.call_args.args[-1].result_size, int(value))

    def test_image_search_requires_a_bounded_nonempty_list_of_string_ids(self):
        for value in (None, "image-1", [], [""], [1], [{}]):
            self.assert_bad_request(self.post("/search/image", id=value), "id")
        self.assert_bad_request(self.post("/search/image", id=["a"] * (limits.MAX_IMAGE_SEARCH_IDS + 1)), "at most")
        self.usecase.search_image.assert_not_called()
        self.assertEqual(self.post("/search/image", id=["image-1"]).status_code, 200)
        self.assertEqual(self.usecase.search_image.call_args.args[1], [ImageId("image-1")])

    def test_text_that_cannot_be_encoded_is_rejected_before_the_usecase(self):
        # "\ud800" in JSON becomes a lone surrogate, which neither SQLite nor a tokenizer accepts.
        for endpoint, body in (
            ("/search/image", '{"params": {"model_name": "ViT-B-32", "id": ["\\ud800"]}}'),
            ("/search/text", '{"params": {"model_name": "ViT-B-32", "text": "cat \\udfff"}}'),
            ("/search/random", '{"params": {"model_name": "\\ud800"}}'),
            ("/downloads/prepare", '{"params": {"ids": ["valid", "\\ud800"]}}'),
        ):
            with self.subTest(endpoint=endpoint):
                response = self.client.post(endpoint, data=body, content_type="application/json")
                self.assert_bad_request(response, "valid Unicode")
                self.assertEqual(response.get_json()["error_code"], "invalid_input")
        self.assertEqual(self.usecase.mock_calls, [])
        self.assertEqual(self.post("/search/text", text="猫 🐈").status_code, 200)

    def test_query_accepts_saved_queries_and_rejects_anything_else(self):
        self.assertEqual(self.post("/search/query", search_query=self.saved_query.to_text()).status_code, 200)
        self.usecase.search_query.assert_called_once_with(MODEL, self.saved_query, SearchOptions())
        valid = self.saved_query.to_text()
        for query in (
            None, "", "[]", "garbage", "[0.1, -0.2]", "{}", valid.replace('"version":1', '"version":2'),
            valid.replace("-0.2", "NaN"), valid.replace("-0.2", "1e99"), valid.replace("-0.2", "true"),
        ):
            with self.subTest(query=query):
                response = self.post("/search/query", search_query=query)
                self.assert_bad_request(response, "search_query")
                self.assertEqual(response.get_json()["error_code"], "invalid_query")
        self.assertEqual(self.usecase.search_query.call_count, 1)

    def test_feature_strength_uses_slider_range(self):
        query = self.saved_query.to_text()
        for strength in (-2, 0, 2):
            response = self.post("/search/queryaddtext", text="cat", search_query=query, features_strength=strength)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(self.usecase.add_text_features.call_args.args[:4], (MODEL, "cat", self.saved_query, strength))
        for strength in (None, True, -2.1, 2.1, float("inf"), "1"):
            response = self.post("/search/queryaddtext", text="cat", search_query=query, features_strength=strength)
            self.assert_bad_request(response, "features_strength")

    def test_upload_rejects_invalid_encoding_and_passes_the_decoded_image(self):
        for value in (None, "", "@@@", "data:image/png;base64,AAAA"):
            self.assert_bad_request(self.post("/search/uploadimage", base64=value), "base64")
        encoded = base64.b64encode(b"bytes").decode("ascii")
        self.assert_bad_request(self.post("/search/uploadimage", base64=encoded, content_type="text/html"), "content_type")
        self.usecase.search_upload_image.assert_not_called()
        binary = BytesIO()
        Image.new("RGB", (1, 1)).save(binary, format="PNG")
        encoded = base64.b64encode(binary.getvalue()).decode("ascii")
        self.assertEqual(self.post("/search/uploadimage", base64=encoded, content_type="image/png").status_code, 200)
        uploaded = self.usecase.search_upload_image.call_args.args[1]
        self.assertEqual((uploaded.binary, uploaded.content_type), (binary.getvalue(), "image/png"))

    def test_usecase_failures_are_reported_with_their_status_and_code(self):
        for error, status, code in (
            (SearchInputError("画像を読み込めません。"), 400, "invalid_model_or_image"),
            (SearchPatternError("正規表現の処理時間を超えました。"), 400, "invalid_pattern"),
            (InvalidSearchQuery("dimension mismatch"), 400, "invalid_query"),
            (ImageUnavailableError("画像IDが見つかりません: a"), 404, "image_unavailable"),
            (ResourceLimitError("検索画像は6400万画素以内で指定してください。"), 413, "resource_limit"),
        ):
            with self.subTest(error=type(error).__name__):
                self.usecase.search_text.side_effect = error
                response = self.post("/search/text", text="cat")
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.get_json(), {"error": str(error), "error_code": code})

    def test_internal_value_errors_are_not_mislabelled_as_client_errors(self):
        self.usecase.search_text.side_effect = ValueError("index is inconsistent")
        with patch.dict(self.app.config, {"TESTING": True}):
            with self.assertRaisesRegex(ValueError, "index is inconsistent"):
                self.post("/search/text", text="cat")

    def test_request_body_limit_returns_json_413(self):
        with patch.dict(self.app.config, {"MAX_CONTENT_LENGTH": 32}):
            response = self.post("/search/text", text="cat" * 100)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error_code"], "request_too_large")

    def test_json_responses_have_correct_mime_and_nosniff(self):
        self.usecase.get_all_model.return_value = []
        self.usecase.get_image_items_by_page.return_value = []
        for response in (
            self.post("/search/text", text="cat"),
            self.client.get("/image_item?model_name=ViT-B-32"), self.client.get("/model_item"),
            self.post("/search/text"),
        ):
            self.assertEqual(response.mimetype, "application/json")
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_start_app_serves_the_usecase_on_the_requested_address(self):
        with patch.object(controller, "configure_logging"), patch.object(controller, "create_app") as factory:
            controller.start_app(self.usecase, host="127.0.0.1", port=8080)
        factory.assert_called_once_with(self.usecase)
        factory.return_value.run.assert_called_once_with(debug=False, port=8080, host="127.0.0.1")


if __name__ == "__main__":
    unittest.main()
