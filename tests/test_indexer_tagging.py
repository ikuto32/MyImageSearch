import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from indexer.tagging import ImageTaggingService, WdTagger, Z3DTagger, load_sidecar_tags, prepare_tag_input


class FakeSession:
    """Returns one prepared score row per input image."""

    def __init__(self, scores, input_shape=(None, 448, 448, 3)):
        self.scores = np.array(scores, dtype=np.float32)
        self.input = types.SimpleNamespace(name="input", shape=input_shape)
        self.batch_sizes = []

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [types.SimpleNamespace(name="output")]

    def run(self, _outputs, feed):
        count = len(feed["input"])
        start = sum(self.batch_sizes)
        self.batch_sizes.append(count)
        return [self.scores[start:start + count]]


def wd_tagger(scores) -> WdTagger:
    tagger = WdTagger.__new__(WdTagger)
    tagger._load_labels([
        {"name": "general", "category": "9"}, {"name": "explicit", "category": "9"},
        {"name": "long_hair", "category": "0"}, {"name": "^_^", "category": "0"},
        {"name": "blue_sky", "category": "0"}, {"name": "hatsune_miku", "category": "4"},
    ])
    tagger.session = FakeSession(scores)
    return tagger


def z3d_tagger(scores, **session_options) -> Z3DTagger:
    tagger = Z3DTagger.__new__(Z3DTagger)
    tagger.threshold = 0.5
    tagger._load_labels([
        {"name": " blue_sky ", "category": "0"}, {"name": "canine", "category": "5"},
        {"name": "artist_name", "category": "1"}, {"name": "krystal", "category": "4"},
        {"name": "outside", "category": "3"},
    ])
    tagger.session = FakeSession(scores, **session_options)
    return tagger


IMAGES = np.zeros((2, 4, 4, 3), dtype=np.float32)


class TagPreprocessingTests(unittest.TestCase):
    def test_prepare_tag_input_consistency(self):
        target_size = 16
        images = [Image.new("RGB", (8, 12), color="red"), Image.new("RGB", (20, 10), color="blue")]

        prepared = [prepare_tag_input(image, target_size) for image in images]
        stacked = np.stack(prepared)

        self.assertEqual(stacked.shape, (len(images), target_size, target_size, 3))
        for single, batch in zip(prepared, stacked):
            np.testing.assert_array_equal(single, batch)

    def test_prepare_tag_input_bgr_and_padding(self):
        target_size = 8
        for mode in ("RGB", "RGBA"):
            image = Image.new(mode, (4, 6), color=(10, 20, 30))
            arr = prepare_tag_input(image, target_size)

            self.assertEqual(arr.shape, (target_size, target_size, 3))
            self.assertEqual(arr.dtype, np.float32)
            self.assertTrue(arr.flags.c_contiguous)
            center = arr[target_size // 2, target_size // 2]
            np.testing.assert_array_equal(center, np.array([30, 20, 10], dtype=np.float32))
            np.testing.assert_array_equal(arr[0, 0], np.array([255.0, 255.0, 255.0]))


class TaggerTests(unittest.TestCase):
    def test_wd_tagger_selects_top_rating_and_tags_above_each_threshold(self):
        tagger = wd_tagger([
            [0.2, 0.7, 0.9, 0.36, 0.35, 0.86],
            [0.6, 0.6, 0.1, 0.1, 0.1, 0.5],
        ])
        results = tagger.predict_batch(IMAGES, general_threshold=0.35, character_threshold=0.85)
        self.assertEqual(results, [
            # float32(0.35) is below 0.35, so "blue sky" is not above the threshold.
            ("explicit", ["long hair", "^_^"], ["hatsune miku"]),
            # Ties keep the first rating.
            ("general", [], []),
        ])
        self.assertEqual(tagger.predict_batch(IMAGES[:0], general_threshold=0.35, character_threshold=0.85), [])

    def test_threshold_comparison_uses_the_unrounded_threshold(self):
        # float32(0.85) is slightly larger than 0.85 and therefore above the threshold.
        tagger = wd_tagger([[1, 0, 0, 0, 0, np.float32(0.85)]])
        self.assertEqual(
            tagger.predict_batch(IMAGES[:1], general_threshold=0.35, character_threshold=0.85)[0][2],
            ["hatsune miku"],
        )

    def test_z3d_tagger_separates_general_and_character_categories(self):
        tagger = z3d_tagger([[0.9, 0.6, 0.99, 0.7, 0.5], [0.1, 0.1, 0.1, 0.1, 0.51]])
        self.assertEqual(tagger.predict_batch(IMAGES), [(["blue sky", "canine"], ["krystal"]), (["outside"], [])])
        self.assertEqual(tagger.session.batch_sizes, [2])

    def test_z3d_model_with_fixed_batch_size_is_run_one_image_at_a_time(self):
        tagger = z3d_tagger([[0.9, 0, 0, 0, 0], [0, 0.9, 0, 0, 0]], input_shape=(1, 448, 448, 3))
        self.assertEqual(tagger.predict_batch(IMAGES), [(["blue sky"], []), (["canine"], [])])
        self.assertEqual(tagger.session.batch_sizes, [1, 1])

    def test_labels_without_a_model_output_are_ignored(self):
        tagger = z3d_tagger([[0.9, 0.9, 0.9, 0.9]])
        self.assertEqual(tagger.predict_batch(IMAGES[:1]), [(["blue sky", "canine"], ["krystal"])])

    def test_service_merges_both_taggers_in_a_stable_order_without_duplicates(self):
        service = ImageTaggingService.__new__(ImageTaggingService)
        service.general_threshold, service.character_threshold = 0.35, 0.85
        service.wd_tagger = wd_tagger([[0.2, 0.7, 0.9, 0.1, 0.8, 0.9], [0.9, 0.1, 0, 0, 0, 0]])
        service.z3d_tagger = z3d_tagger([[0.9, 0.6, 0, 0.7, 0], [0, 0, 0, 0, 0]])
        self.assertEqual(service.tag_batch(IMAGES, IMAGES), [
            {"rating": "explicit", "tags": [
                "long hair", "blue sky", "canine", "character:hatsune miku", "character:krystal",
            ]},
            {"rating": "general", "tags": []},
        ])


class SidecarTagsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = Path("nested/photo.jpg")
        self.sidecar = self.root / "nested/photo.jpg.tags.json"
        self.sidecar.parent.mkdir()

    def load(self):
        return load_sidecar_tags(str(self.root), str(self.image))

    def test_absent_sidecar_leaves_image_unclassified_without_warning(self):
        with mock.patch("builtins.print") as warning:
            self.assertEqual(self.load(), {"rating": "", "tags": []})
        warning.assert_not_called()

    def test_nested_sidecar_preserves_unicode_tags_and_optional_fields(self):
        for data in ({"rating": "general", "tags": ["山", "sky"]}, {"tags": ["snow"]}, {}):
            with self.subTest(data=data):
                self.sidecar.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8-sig")
                self.assertEqual(self.load(), {"rating": data.get("rating", ""), "tags": data.get("tags", [])})

    def test_invalid_sidecar_warns_and_does_not_block_image_indexing(self):
        contents = (b"{", b"\xff", b"[]", b'{"rating": 1}', b'{"tags": "sky"}', b'{"tags": [1]}')
        for content in contents:
            with self.subTest(content=content), mock.patch("builtins.print") as warning:
                self.sidecar.write_bytes(content)
                self.assertEqual(self.load(), {"rating": "", "tags": []})
                self.assertIn(str(self.sidecar), warning.call_args.args[0])

    def test_unreadable_sidecar_warns_and_leaves_image_unclassified(self):
        with mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")), mock.patch("builtins.print") as warning:
            self.assertEqual(self.load(), {"rating": "", "tags": []})
        self.assertIn("denied", warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
