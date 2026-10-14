"""APIパラメータの取り出しと検証。

インデックスやモデルを読み込む前に、利用者の入力の誤りを400として報告する。
"""
import base64
import binascii
import json
import math
import re

from flask import request

from app.application.search import AESTHETIC_QUALITY_MAX, AESTHETIC_QUALITY_MIN, SearchOptions
from app.application.search_index import AESTHETIC_MODEL_NAMES
from app.application.text_matching import SearchMatcher
from app.domain import limits
from app.domain.domain_object import Image, ImageId, ModelId
from app.domain.rating import RATING_CATEGORIES
from app.domain.search_query import SearchQuery, parse_search_query

DEFAULT_IMAGE_ITEM_PAGE_SIZE = 60
MAX_IMAGE_ITEM_PAGE_SIZE = 240
MAX_RATINGS_QUERY_LENGTH = 256


class InputValidationError(ValueError):
    """Invalid API input, distinct from failures inside the use case."""


def json_params() -> dict:
    """POSTされたJSON bodyのparamsからAPIパラメータを取り出す"""
    try:
        body = request.get_json(silent=True)
    except RecursionError:
        raise InputValidationError("Request JSON is nested too deeply") from None
    if not isinstance(body, dict):
        raise InputValidationError("Request body must be a JSON object")
    params = body.get("params")
    if not isinstance(params, dict):
        raise InputValidationError("params must be a JSON object")
    return params


def integer_parameter(value, name: str, minimum: int, maximum: int | None = None) -> int:
    # Query strings supply integers as strings.
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value):
        try:
            value = int(value)
        except ValueError:
            raise InputValidationError(f"{name} must be an integer") from None
    if type(value) is not int:
        raise InputValidationError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        limit = f"{minimum} to {maximum}" if maximum is not None else f"at least {minimum}"
        raise InputValidationError(f"{name} must be {limit}")
    return value


def string_parameter(value, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise InputValidationError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")
    # JSONの "\ud800" のような対にならないサロゲートは、SQLiteやモデルに渡せない。
    # ASCIIだけの大きなbase64は、複製を作らずに通す。
    if not value.isascii():
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise InputValidationError(f"{name} must be valid Unicode text") from None
    return value


def number_parameter(value, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float):
        raise InputValidationError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or not minimum <= value <= maximum:
        raise InputValidationError(f"{name} must be a finite number from {minimum} to {maximum}")
    return float(value)


def boolean_parameter(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise InputValidationError(f"{name} must be true or false")
    return value


def model_id_parameter(params) -> ModelId:
    model_name = string_parameter(params.get("model_name"), "model_name")
    pretrained = params.get("pretrained")
    if pretrained is None:
        pretrained = ""
    return ModelId(model_name, string_parameter(pretrained, "pretrained", allow_empty=True))


def search_options(params: dict) -> SearchOptions:
    """Validate shared search settings before starting any index/model work."""
    beta = number_parameter(params.get("aesthetic_quality_beta", 0), "aesthetic_quality_beta", -1, 1)
    quality_range = params.get("aesthetic_quality_range", [AESTHETIC_QUALITY_MIN, AESTHETIC_QUALITY_MAX])
    if not isinstance(quality_range, list) or len(quality_range) != 2:
        raise InputValidationError("aesthetic_quality_range must contain two numbers")
    low, high = (
        number_parameter(value, "aesthetic_quality_range", AESTHETIC_QUALITY_MIN, AESTHETIC_QUALITY_MAX)
        for value in quality_range
    )
    if low > high:
        raise InputValidationError("aesthetic_quality_range minimum must not exceed maximum")
    model_name = params.get("aesthetic_model_name")
    if model_name is None:
        model_name = AESTHETIC_MODEL_NAMES[0]
    if string_parameter(model_name, "aesthetic_model_name") not in AESTHETIC_MODEL_NAMES:
        raise InputValidationError("aesthetic_model_name must be " + " or ".join(AESTHETIC_MODEL_NAMES))
    result_size = integer_parameter(
        params.get("result_size", limits.DEFAULT_RESULT_SIZE), "result_size", 1, limits.MAX_RESULT_SIZE
    )
    return SearchOptions(beta, (low, high), model_name, result_size)


def text_parameter(params: dict) -> str:
    return string_parameter(params.get("text"), "text")


def pattern_parameters(params: dict) -> tuple[str, bool]:
    """文字列検索の語と、それを正規表現として扱うか。式の誤りもここで報告する。"""
    text = text_parameter(params)
    is_regexp = boolean_parameter(params.get("is_regexp", False), "is_regexp")
    SearchMatcher(text, is_regexp)
    return text, is_regexp


def image_ids_parameter(value, name: str, maximum: int) -> list[ImageId]:
    if not isinstance(value, list) or not value:
        raise InputValidationError(f"{name} must be a non-empty list of image IDs")
    if len(value) > maximum:
        raise InputValidationError(f"{name} must contain at most {maximum} image IDs")
    return [ImageId(string_parameter(item, name)) for item in value]


def ratings_parameter(value) -> list[str] | None:
    if value is not None and (
        not isinstance(value, list) or len(value) > len(RATING_CATEGORIES)
        or any(not isinstance(rating, str) or rating not in RATING_CATEGORIES for rating in value)
    ):
        raise InputValidationError("ratings must be a list of supported image categories")
    return value


def ratings_query_parameter(text: str | None) -> list[str] | None:
    """query stringにJSON配列として指定された画像区分。未指定はNone。"""
    if text is None:
        return None
    try:
        if len(text) > MAX_RATINGS_QUERY_LENGTH:
            raise ValueError()
        ratings = json.loads(text)
    except (ValueError, TypeError, RecursionError):
        raise InputValidationError("ratings must be a JSON array of supported image categories") from None
    if not isinstance(ratings, list):
        raise InputValidationError("ratings must be a JSON array of supported image categories")
    return ratings_parameter(ratings)


def search_query_parameter(params: dict) -> SearchQuery:
    return parse_search_query(params.get("search_query"))


def uploaded_image_parameter(params: dict) -> Image:
    encoded = string_parameter(params.get("base64"), "base64")
    content_type = string_parameter(params.get("content_type", "image/png"), "content_type")
    if not content_type.startswith("image/"):
        raise InputValidationError("content_type must be an image MIME type")
    try:
        binary = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise InputValidationError("base64 must contain a valid encoded image") from None
    return Image(binary, content_type)
