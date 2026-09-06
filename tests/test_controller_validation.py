import base64
from io import BytesIO
import unittest
from unittest.mock import Mock, patch
import tempfile

import numpy as np
from PIL import Image

from app.domain.domain_object import ImageId, ModelId, ResultImageItemList
from app.domain.errors import ResourceLimitError
from app.presentation import controller


class ControllerValidationTests(unittest.TestCase):
    def setUp(self):
        self.usecase = Mock()
        self.usecase.get_image_items_by_page.return_value = []
        for method in (
            "search_text", "search_name", "search_image", "search_random",
            "search_tags", "search_style_cluster", "search_query",
            "add_text_features", "search_upload_image",
        ):
            getattr(self.usecase, method).return_value = ResultImageItemList([], "")
        self.usecase_patch = patch.object(controller, "usecase", self.usecase)
        self.usecase_patch.start()
        self.addCleanup(self.usecase_patch.stop)
        self.client = controller.app.test_client()

    def post(self, endpoint, **params):
        return self.client.post(endpoint, json={"params": {"model_name": "ViT-B-32", **params}})

    def assert_bad_request(self, response, parameter=None):
        self.assertEqual(response.status_code, 400, response.data)
        self.assertTrue(response.is_json)
        error = response.get_json().get("error")
        self.assertIsInstance(error, str)
        if parameter:
            self.assertIn(parameter, error)

    def test_all_post_routes_reject_invalid_json_objects(self):
        for endpoint in (
            "/search/text", "/search/name", "/search/image", "/search/random",
            "/search/tags", "/search/style_cluster", "/search/query",
            "/search/queryaddtext", "/search/uploadimage", "/image_ratings",
            "/image_meta/image-1", "/download_images_zip",
        ):
            for body in ('{', 'null', '[]', '"text"', '{"params": null}', '{"params": []}'):
                with self.subTest(endpoint=endpoint, body=body):
                    response = self.client.post(endpoint, data=body, content_type="application/json")
                    self.assert_bad_request(response)
        self.assertEqual(self.usecase.mock_calls, [])

    def test_native_json_boolean_and_null_optional_model_fields_work(self):
        response = self.post(
            "/search/name", text="cat", is_regexp=True, pretrained=None,
            aesthetic_model_name=None,
        )
        self.assertEqual(response.status_code, 200)
        args = self.usecase.search_name.call_args.args
        self.assertEqual(args[0], ModelId("ViT-B-32", ""))
        self.assertEqual(args[1].text, "cat")
        self.assertEqual(args[2:], (True, 0.0, 0.0, 10.0, "original", 2048))

    def test_regex_searches_accept_native_and_legacy_booleans(self):
        for endpoint, method in (
            ("/search/name", "search_name"), ("/search/tags", "search_tags"),
            ("/search/style_cluster", "search_style_cluster"),
        ):
            for supplied, expected in ((True, True), (False, False), ("true", True), ("false", False)):
                with self.subTest(endpoint=endpoint, supplied=supplied):
                    response = self.post(endpoint, text="cat", is_regexp=supplied)
                    self.assertEqual(response.status_code, 200)
                    self.assertIs(getattr(self.usecase, method).call_args.args[2], expected)

    def test_search_text_and_random_supply_shared_defaults(self):
        self.assertEqual(self.post("/search/text", text="cat").status_code, 200)
        self.assertEqual(self.usecase.search_text.call_args.args[2:], (0.0, 0.0, 10.0, "original", 2048))
        self.assertEqual(self.post("/search/random").status_code, 200)
        self.assertEqual(self.usecase.search_random.call_args.args[1:], (0.0, 0.0, 10.0, "original", 2048))

    def test_flat_json_params_remain_supported(self):
        response = self.client.post("/search/text", json={"model_name": "ViT-B-32", "text": "cat"})
        self.assertEqual(response.status_code, 200)

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

    def test_supported_aesthetic_models_are_passed_through(self):
        for model_name in ("original", "pony"):
            self.assertEqual(self.post("/search/random", aesthetic_model_name=model_name).status_code, 200)
            self.assertEqual(self.usecase.search_random.call_args.args[-2], model_name)

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

    def test_invalid_booleans_and_regex_are_reported_as_input_errors(self):
        for value in (None, 0, 1, "yes", [], {}):
            self.assert_bad_request(self.post("/search/name", text="cat", is_regexp=value), "is_regexp")
        self.assert_bad_request(self.post("/search/name", text="[", is_regexp=True), "正規表現")
        self.assert_bad_request(self.post("/search/name", text="a{999999999999999999999}", is_regexp=True), "正規表現")
        self.usecase.search_name.assert_not_called()

    def test_result_size_rejects_invalid_values_and_accepts_limit(self):
        for value in (None, True, 0, -1, 1.5, "1.5", "invalid", controller.MAX_RESULT_SIZE + 1):
            with self.subTest(value=value):
                self.assert_bad_request(self.post("/search/random", result_size=value), "result_size")
        for value in (1, "60", controller.MAX_RESULT_SIZE):
            self.assertEqual(self.post("/search/random", result_size=value).status_code, 200)
            self.assertEqual(self.usecase.search_random.call_args.args[-1], int(value))

    def test_image_pagination_is_bounded_and_validated(self):
        self.assertEqual(self.client.get("/image_item").status_code, 200)
        self.usecase.get_image_items_by_page.assert_called_with(0, 60)
        self.assertEqual(self.client.get("/image_item?page=2&size=240").status_code, 200)
        self.usecase.get_image_items_by_page.assert_called_with(2, 240)
        self.usecase.reset_mock()
        for query in ("page=-1", "page=x", "page=1.5", "size=0", "size=-1", "size=241", "size=x"):
            with self.subTest(query=query):
                self.assert_bad_request(self.client.get(f"/image_item?{query}"))
        self.usecase.get_image_items_by_page.assert_not_called()

    def test_image_search_requires_a_nonempty_list_of_string_ids(self):
        for value in (None, "image-1", [], [""], [1], [{}]):
            self.assert_bad_request(self.post("/search/image", id=value), "id")
        self.assertEqual(self.post("/search/image", id=["image-1"]).status_code, 200)
        self.assertEqual(self.usecase.search_image.call_args.args[1], [ImageId("image-1")])

    def test_query_accepts_saved_one_row_vectors_and_rejects_invalid_numbers(self):
        for query in ("[[0.1, -0.2, 1e-5]]", "[0.1, -0.2, 1e-5]", "0.1, -0.2, 1e-5"):
            self.assertEqual(self.post("/search/query", search_query=query).status_code, 200)
            self.assertEqual(self.usecase.search_query.call_args.args[1], "[0.1, -0.2, 1e-05]")
        for query in (None, "", "[]", "[[]]", "garbage", "[1, rubbish]", "[NaN]", "[1e99]", "[true]", "[[1], [2]]"):
            with self.subTest(query=query):
                self.assert_bad_request(self.post("/search/query", search_query=query), "search_query")

    def test_query_accepts_actual_numpy_formatted_search_response(self):
        usecase = controller.Usecase.__new__(controller.Usecase)
        # Normalization is unrelated to NumPy's non-JSON output syntax.
        with patch("app.application.usecase.faiss.normalize_L2", create=True):
            query = usecase.format_search_query(np.array([[1., 0., -0.]], dtype=np.float32))
        self.assertIn("1.", query)
        self.assertEqual(self.post("/search/query", search_query=query).status_code, 200)
        self.assertEqual(self.post("/search/queryaddtext", text="cat", search_query=query).status_code, 200)
        self.assertEqual(self.post("/search/query", search_query="[.1, 1., -0., 1e-3]").status_code, 200)

    def test_feature_strength_uses_slider_range(self):
        for strength in (-2, 0, 2):
            self.assertEqual(self.post("/search/queryaddtext", text="cat", search_query="[1,2]", features_strength=strength).status_code, 200)
            self.assertEqual(self.usecase.add_text_features.call_args.args[3], strength)
        for strength in (None, True, -2.1, 2.1, float("inf"), "1"):
            self.assert_bad_request(self.post("/search/queryaddtext", text="cat", search_query="[1,2]", features_strength=strength), "features_strength")

    def test_upload_rejects_invalid_encoding_and_nonimages(self):
        for value in (None, "", "@@@", "!!!!,@@@@", "SGVsbG8="):
            self.assert_bad_request(self.post("/search/uploadimage", base64=value), "base64")
        self.usecase.search_upload_image.assert_not_called()
        binary = BytesIO()
        Image.new("RGB", (1, 1)).save(binary, format="PNG")
        encoded = base64.b64encode(binary.getvalue()).decode("ascii")
        self.assertEqual(self.post("/search/uploadimage", base64=f"data:image/png;base64,{encoded}").status_code, 200)
        self.assertEqual(self.usecase.search_upload_image.call_args.args[1].binary, binary.getvalue())

    def test_internal_value_errors_are_not_mislabelled_as_client_errors(self):
        self.usecase.search_text.side_effect = ValueError("index is inconsistent")
        with patch.dict(controller.app.config, {"TESTING": True, "PROPAGATE_EXCEPTIONS": True}):
            with self.assertRaisesRegex(ValueError, "index is inconsistent"):
                self.post("/search/text", text="cat")

    def test_start_app_accepts_explicit_bind_address(self):
        with patch.object(controller, "configure_logging"), patch.object(controller.app, "run") as run:
            controller.start_app(self.usecase, host="127.0.0.1", port=8080)
        run.assert_called_once_with(debug=False, port=8080, host="127.0.0.1")

    def test_json_responses_have_correct_mime_and_nosniff(self):
        self.usecase.get_image_metadata.return_value = {"tags": "<script>alert(1)</script>"}
        self.usecase.get_rating_list.return_value = {}
        self.usecase.get_all_model.return_value = []
        for response in (
            self.post("/search/text", text="cat"),
            self.client.get("/image_item"), self.client.get("/model_item"),
            self.post("/image_meta/image-1"), self.post("/image_ratings", ids=[]),
        ):
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, "application/json")
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_id_batch_limits_are_checked_before_usecase(self):
        for endpoint, parameter, limit in (
            ("/search/image", "id", controller.MAX_IMAGE_SEARCH_IDS),
            ("/image_ratings", "ids", controller.MAX_RATING_IDS),
            ("/download_images_zip", "ids", controller.MAX_ZIP_IDS),
        ):
            self.assert_bad_request(self.post(endpoint, **{parameter: ["a"] * (limit + 1)}), "at most")
        self.assertEqual(self.usecase.mock_calls, [])

    def test_ratings_requires_explicit_ids_and_empty_list_is_empty(self):
        self.assert_bad_request(self.post("/image_ratings"), "ids is required")
        self.assert_bad_request(self.client.get("/image_ratings?model_name=ViT-B-32"), "ids is required")
        self.usecase.get_rating_list.return_value = {}
        response = self.post("/image_ratings", ids=[])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {})
        self.usecase.get_rating_list.assert_called_once_with(ModelId("ViT-B-32", ""), [])

    def test_request_body_limit_and_image_resource_limits_return_json_413(self):
        with patch.dict(controller.app.config, {"MAX_CONTENT_LENGTH": 32}):
            response = self.post("/search/text", text="cat" * 100)
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.is_json)
        self.usecase.get_images_zip.side_effect = ResourceLimitError("archive exceeds limit")
        response = self.post("/download_images_zip", ids=["a"])
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error"], "archive exceeds limit")

    def test_deeply_nested_json_returns_400(self):
        response = self.client.post("/search/random", data='{"params":' + '[' * 2000 + '0' + ']' * 2000 + '}', content_type="application/json")
        self.assert_bad_request(response)

    def test_download_closes_spooled_file_after_stream_and_send_failure(self):
        for buffered in (False, True):
            spool = tempfile.SpooledTemporaryFile(max_size=1, mode="w+b")
            spool.write(b"zip test payload")
            spool.seek(0)
            self.usecase.get_images_zip.return_value = spool
            response = self.client.post("/download_images_zip", json={"params": {"ids": ["a"]}}, buffered=buffered)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.data, b"zip test payload")
            response.close()
            self.assertTrue(spool.closed)
        spool = tempfile.SpooledTemporaryFile()
        self.usecase.get_images_zip.return_value = spool
        with patch.object(controller, "send_file", side_effect=OSError("send failed")):
            with controller.app.test_request_context("/download_images_zip", method="POST", json={"params": {"ids": ["a"]}}):
                with self.assertRaises(OSError):
                    controller.download_images_zip()
        self.assertTrue(spool.closed)


if __name__ == "__main__":
    unittest.main()
