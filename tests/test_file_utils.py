import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.presentation.file_utils import (
    ALLOWED_IMAGE,
    ALLOWED_VIDEO,
    ALLOWED_TEXT,
    file_category,
)


def test_file_category_image():
    for ext in ALLOWED_IMAGE:
        assert file_category(f"dummy.{ext}") == "image"
        assert file_category(f"dummy.{ext.upper()}") == "image"


def test_file_category_video():
    for ext in ALLOWED_VIDEO:
        assert file_category(f"movie.{ext}") == "video"
        assert file_category(f"movie.{ext.upper()}") == "video"


def test_file_category_text():
    for ext in ALLOWED_TEXT:
        assert file_category(f"file.{ext}") == "text"
        assert file_category(f"file.{ext.upper()}") == "text"


def test_file_category_unknown():
    assert file_category("example.bin") is None
